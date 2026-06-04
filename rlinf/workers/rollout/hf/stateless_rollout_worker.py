# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import gc
from typing import Any, Literal

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from rlinf.config import SupportedModel
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.scheduler import Channel, Cluster, CollectiveGroupOptions, Worker
from rlinf.utils.placement import HybridComponentPlacement


class StatelessRolloutWorker(Worker):
    """Stateless rollout (model-forward) server.

    The worker holds the policy weights and answers `predict` requests. It does
    not own per-episode state, env-side rank maps, or chunk loop counters; those
    live in `EmbodiedLoopWorker`. Weight sync from actor stays here.
    """

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg

        self.actor_group_name = cfg.actor.group_name
        self.device = self.torch_platform.current_device()
        self.placement = HybridComponentPlacement(cfg, Cluster())

        actor_world_size = self.placement.get_world_size("actor")
        self.actor_weight_src_rank = self._rank % actor_world_size

        self.enable_offload = self.cfg.rollout.get("enable_offload", False)
        self.enable_cuda_graph = self.cfg.rollout.get("enable_cuda_graph", False)

        self.expert_model = None

        # Sync weight comm options
        max_ctas = cfg.rollout.get("sync_weight_nccl_max_ctas", None)
        min_ctas = cfg.rollout.get("sync_weight_nccl_min_ctas", None)
        self._sync_weight_comm_options = CollectiveGroupOptions(
            accel_max_ctas=max_ctas, accel_min_ctas=min_ctas
        )

    def init_worker(self):
        rollout_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.model_path = self.cfg.rollout.model.model_path

        self.hf_model: BasePolicy = get_model(rollout_model_config)

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            self.hf_model.load_state_dict(model_dict)

        if self.cfg.rollout.get("expert_model", None):
            expert_model_config = copy.deepcopy(self.cfg.actor.model)
            with open_dict(expert_model_config):
                expert_model_config.precision = self.cfg.rollout.expert_model.precision
                expert_model_config.model_path = (
                    self.cfg.rollout.expert_model.model_path
                )
            self.expert_model = get_model(expert_model_config)

            if self.cfg.runner.get("expert_ckpt_path", None):
                expert_model_dict = torch.load(self.cfg.runner.expert_ckpt_path)
                self.expert_model.load_state_dict(expert_model_dict)

        self.hf_model.eval()
        if self.expert_model is not None:
            self.expert_model.eval()

        if self.cfg.rollout.get("enable_torch_compile", False):
            mode = self.cfg.rollout.get(
                "torch_compile_mode", "max-autotune-no-cudagraphs"
            )
            self.hf_model.enable_torch_compile(mode=mode)
        if self.enable_cuda_graph and not self.enable_offload:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size,
            )

        self.setup_sample_params()
        if self.enable_offload:
            self.offload_model()

    def setup_sample_params(self):
        # length parameters for rollout
        self._length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        # sampling parameters for rollout
        self._sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )
        self._train_sampling_params = {
            "do_sample": self._sampling_params["do_sample"],
            "temperature": self._sampling_params["temperature_train"]
            if self._sampling_params["do_sample"]
            else 1.0,
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

        self._eval_sampling_params = {
            "do_sample": True
            if self._sampling_params.get("temperature_eval", -1) > 0
            else False,
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

    @Worker.timer("rollout_predict")
    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "eval",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
            SupportedModel.DREAMZERO,
            SupportedModel.CNN_POLICY,
        ]:
            if self.cfg.algorithm.loss_type == "embodied_dagger":
                kwargs = {"mode": "eval"}
            else:
                kwargs = {"mode": mode}

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.CNN_POLICY,
            SupportedModel.FLOW_POLICY,
            SupportedModel.MLP_POLICY,
        ]:
            kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        with torch.no_grad():
            actions, result = self.hf_model.predict_action_batch(
                env_obs=env_obs,
                **kwargs,
            )

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        result["expert_label_flag"] = False
        return actions, result

    async def evaluate_start(
        self,
        request_channel: Channel,
        response_channel: Channel,
        mode: Literal["train", "eval"] = "eval",
    ):
        """Stateless forward server: serve requests until a `_stop` sentinel.

        All rollout workers `.get()` from the request channel's default key in
        FIFO order (no rank pairing). The request payload carries `send_key`
        (chosen by the caller), so replies are routed back via that key.

        Stop:
            `evaluate_stop(request_channel)` is a group RPC; each rank pushes
            one `{"_stop": True}` sentinel onto the request queue. With the
            sentinels matching the N parked `get()`s 1:1, every worker breaks
            its `while True` exactly once.
        """
        if self.enable_offload:
            self.reload_model()

        while True:
            req = await request_channel.get(async_op=True).async_wait()

            if isinstance(req, dict) and req.get("_stop", False):
                self.log_info(
                    f"StatelessRolloutWorker rank={self._rank} got stop sentinel"
                )
                break

            send_key = req["send_key"]
            req_mode = req.get("mode", mode)
            actions, _result = self.predict(req["obs"], mode=req_mode)

            # Eval only needs actions; for train, extend the payload to
            # include `_result` (forward_inputs etc.).
            if isinstance(actions, torch.Tensor):
                actions_payload = actions.detach().cpu().contiguous()
            else:
                actions_payload = actions
            response_channel.put(
                {"actions": actions_payload},
                key=send_key,
                async_op=True,
            )

        if self.enable_offload:
            self.offload_model()

    def evaluate_stop(self, request_channel: Channel):
        """Push one stop sentinel into `request_channel` (group RPC: one per rank)."""
        request_channel.put({"_stop": True}, async_op=False)

    async def sync_model_from_actor(self):
        """Sync model parameters from the actor worker."""
        param_state_dict = await self.recv(
            self.actor_group_name,
            src_rank=self.actor_weight_src_rank,
            async_op=True,
            options=self._sync_weight_comm_options,
        ).async_wait()
        self.hf_model.load_state_dict(param_state_dict)

        del param_state_dict
        gc.collect()
        self.torch_platform.empty_cache()

    def offload_model(self):
        if self.enable_cuda_graph:
            self.hf_model.release_cuda_graph()
        self.hf_model.to("cpu")
        self.torch_platform.empty_cache()

    def reload_model(self):
        self.hf_model.to(self.device)
        if self.enable_cuda_graph:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size,
            )

    def set_global_step(self, global_step: int):
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
