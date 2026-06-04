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

"""EmbodiedLoopWorker — stateful agent-loop driver.

Holds chunk-step counters and rank pairing with `EnvWorker`, but **does not
hold the model**. Forward calls go to a pool of `StatelessRolloutWorker`s over
a shared request/response channel pair — any free rollout worker picks up the
next request (no sticky pairing).

Wire diagram (eval):

    EnvWorker (rank R)
        --[EnvObs chan, key="{R}_{R}_eval_obs"]-->
    EmbodiedLoop (rank R)
        --[RolloutRequest chan, default key,
           payload={"obs":..., "send_key": resp_key}]-->
    StatelessRolloutWorker (any free rank)
        --[RolloutResponse chan, key=resp_key]-->
    EmbodiedLoop (rank R)
        --[EnvAction chan, key="{R}_{R}_eval_actions"]-->
    EnvWorker (rank R)

Placement slots:
    * `embodied_loop` — this worker (stateful agent-loop dispatcher).
                        Must have the same world size as `env`.
    * `rollout`       — `StatelessRolloutWorker` (stateless model forward).
                        World size is independent — the pool is FIFO-shared.

`EnvWorker` reads its peer component name from `cfg.env.peer_component`
(default "rollout"). When using this worker, set it to "embodied_loop" so
env↔loop channel keys align even when the rollout pool has a different size.
"""

from typing import Any, Literal

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.placement import HybridComponentPlacement


class EmbodiedLoopWorker(Worker):
    """Stateful agent-loop / chunk dispatcher.

    Ranks are 1:1 aligned with EnvWorker (placement key "embodied_loop").
    Dispatches forward requests to the StatelessRolloutWorker pool
    (no sticky pairing — any free rank picks up the next request).
    """

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.placement = HybridComponentPlacement(cfg, Cluster())

        self.num_pipeline_stages = cfg.embodied_loop.get(
            "pipeline_stage_num", cfg.rollout.pipeline_stage_num
        )
        self.total_num_train_envs = cfg.env.train.total_num_envs
        self.total_num_eval_envs = cfg.env.eval.total_num_envs

        self.enable_eval = (
            cfg.runner.val_check_interval > 0 or cfg.runner.only_eval
        )

        self.n_train_chunk_steps = (
            cfg.env.train.max_steps_per_rollout_epoch
            // cfg.actor.model.num_action_chunks
        )
        self.n_eval_chunk_steps = (
            cfg.env.eval.max_steps_per_rollout_epoch
            // cfg.actor.model.num_action_chunks
        )

        self.version = 0
        self.finished_episodes = None

    def init_worker(self):
        # Per-loop unique response key — each EmbodiedLoop rank has at most one
        # outstanding rollout request at a time (its evaluate body awaits the
        # reply before issuing the next), so a stable per-rank key gives strict
        # FIFO request/response ordering without a per-request counter.
        self._rollout_resp_key = f"rollout_response_{self._rank}"

        self.dst_ranks: dict[str, list[tuple[int, int]]] = {}
        self.src_ranks: dict[str, list[tuple[int, int]]] = {}
        if not self.cfg.runner.only_eval:
            self.dst_ranks["train"] = self._setup_dst_ranks(
                self.total_num_train_envs // self.num_pipeline_stages
            )
            self.src_ranks["train"] = self._setup_src_ranks(
                self.total_num_train_envs // self.num_pipeline_stages
            )
        if self.enable_eval:
            self.dst_ranks["eval"] = self._setup_dst_ranks(
                self.total_num_eval_envs // self.num_pipeline_stages
            )
            self.src_ranks["eval"] = self._setup_src_ranks(
                self.total_num_eval_envs // self.num_pipeline_stages
            )

        self.log_info(
            f"EmbodiedLoopWorker initialized: dst_ranks={self.dst_ranks}, "
            f"src_ranks={self.src_ranks}, "
            f"rollout_resp_key={self._rollout_resp_key}"
        )

    def _setup_dst_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        env_world_size = self.placement.get_world_size("env")
        loop_world_size = self.placement.get_world_size("embodied_loop")
        return CommMapper.get_dst_ranks(
            batch_size=batch_size,
            src_world_size=loop_world_size,
            dst_world_size=env_world_size,
            src_rank=self._rank,
        )

    def _setup_src_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        env_world_size = self.placement.get_world_size("env")
        loop_world_size = self.placement.get_world_size("embodied_loop")
        return CommMapper.get_src_ranks(
            batch_size=batch_size,
            src_world_size=env_world_size,
            dst_world_size=loop_world_size,
            dst_rank=self._rank,
        )

    # ---- env-side I/O (replicate MultiStepRolloutWorker's helpers) -------

    @staticmethod
    def _infer_env_batch_size(obs_batch: dict[str, Any]) -> int:
        obs = obs_batch["obs"] if "obs" in obs_batch else obs_batch
        for key in ("states", "main_images", "task_descriptions"):
            value = obs.get(key)
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if isinstance(value, list):
                return len(value)
        raise ValueError("Cannot infer batch size from env obs.")

    @staticmethod
    def _merge_obs_batches(obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        if not obs_batches:
            return {}
        obs_dicts = [
            obs_batch["obs"] if "obs" in obs_batch else obs_batch
            for obs_batch in obs_batches
        ]
        final_obs_list = [
            obs_batch.get("final_obs", None) for obs_batch in obs_batches
        ]

        def _merge_obs_dicts(dicts: list[dict[str, Any]]) -> dict[str, Any]:
            merged: dict[str, Any] = {}
            for key in dicts[0].keys():
                values = [obs_dict[key] for obs_dict in dicts]
                first_non_none = next(
                    (value for value in values if value is not None), None
                )
                if first_non_none is None:
                    merged[key] = None
                elif isinstance(first_non_none, torch.Tensor):
                    merged[key] = torch.cat(values, dim=0)
                elif isinstance(first_non_none, list):
                    merged[key] = [item for sublist in values for item in sublist]
                else:
                    merged[key] = values
            return merged

        merged_obs = _merge_obs_dicts(obs_dicts)
        merged_final_obs = None
        if any(final_obs is not None for final_obs in final_obs_list):
            final_obs_or_obs = [
                final_obs if final_obs is not None else obs_dict
                for obs_dict, final_obs in zip(obs_dicts, final_obs_list)
            ]
            merged_final_obs = _merge_obs_dicts(final_obs_or_obs)

        return {"obs": merged_obs, "final_obs": merged_final_obs}

    async def recv_env_output(
        self,
        input_channel: Channel,
        mode: Literal["train", "eval"] = "eval",
    ) -> dict[str, Any]:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_ranks[mode]
        obs_batches = []
        for src_rank, expected_size in src_ranks_and_sizes:
            obs_batch = await input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra=f"{mode}_obs"
                ),
                async_op=True,
            ).async_wait()
            actual_size = self._infer_env_batch_size(obs_batch)
            assert actual_size == expected_size, (
                f"Expected env output batch size {expected_size} from env rank "
                f"{src_rank}, got {actual_size}."
            )
            obs_batches.append(obs_batch)
        return self._merge_obs_batches(obs_batches)

    def _split_actions(
        self, actions: torch.Tensor | np.ndarray, sizes: list[int]
    ) -> list[torch.Tensor | np.ndarray]:
        assert sum(sizes) == actions.shape[0], (
            f"Number of actions ({actions.shape[0]}) must equal split sizes "
            f"sum ({sum(sizes)})."
        )
        if isinstance(actions, np.ndarray):
            split_indices = np.cumsum(sizes[:-1]).tolist()
            return list(np.split(actions, split_indices, axis=0))
        return list(torch.split(actions, sizes, dim=0))

    def send_chunk_actions(
        self,
        output_channel: Channel,
        chunk_actions: torch.Tensor | np.ndarray,
        mode: Literal["train", "eval"] = "eval",
    ):
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        dst_ranks_and_sizes = self.dst_ranks[mode]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        chunk_actions_split = self._split_actions(chunk_actions, split_sizes)
        for (dst_rank, _), chunk_action_i in zip(
            dst_ranks_and_sizes, chunk_actions_split
        ):
            if isinstance(chunk_action_i, torch.Tensor):
                chunk_action_i = chunk_action_i.detach().cpu().contiguous()
            output_channel.put(
                chunk_action_i,
                key=CommMapper.build_channel_key(
                    self._rank, dst_rank, extra=f"{mode}_actions"
                ),
                async_op=True,
            )

    # ---- rollout-side I/O ------------------------------------------------

    async def request_rollout(
        self,
        rollout_req_channel: Channel,
        rollout_resp_channel: Channel,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
    ) -> torch.Tensor:
        """Send obs to the stateless rollout pool, await action chunk.

        Request goes onto the request channel's default queue (any free
        rollout worker picks it up). The reply is routed back via the per-loop
        `_rollout_resp_key`, carried in the request payload as `send_key`.
        """
        rollout_req_channel.put(
            {
                "obs": env_obs,
                "mode": mode,
                "send_key": self._rollout_resp_key,
            },
            async_op=True,
        )
        resp = await rollout_resp_channel.get(
            key=self._rollout_resp_key, async_op=True
        ).async_wait()
        actions = resp["actions"]
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        return actions

    # ---- main loops ------------------------------------------------------

    @Worker.timer("embodied_loop_evaluate")
    async def evaluate(
        self,
        env_input_channel: Channel,
        env_output_channel: Channel,
        rollout_req_channel: Channel,
        rollout_resp_channel: Channel,
    ):
        """Pull obs from env, ask the rollout pool for actions, push actions to env."""
        for _ in tqdm(
            range(self.cfg.algorithm.eval_rollout_epoch),
            desc="EmbodiedLoop eval epochs",
            disable=(self._rank != 0),
        ):
            for _ in range(self.n_eval_chunk_steps):
                for _ in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(
                        env_input_channel, mode="eval"
                    )
                    actions = await self.request_rollout(
                        rollout_req_channel,
                        rollout_resp_channel,
                        env_output["obs"],
                        mode="eval",
                    )
                    self.send_chunk_actions(
                        env_output_channel, actions, mode="eval"
                    )

    def set_global_step(self, global_step: int):
        self.version = global_step
        if self.finished_episodes is None:
            self.finished_episodes = (
                self.version
                * self.total_num_train_envs
                * self.cfg.algorithm.get("rollout_epoch", 1)
            )
