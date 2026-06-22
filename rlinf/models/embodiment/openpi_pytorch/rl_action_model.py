# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RL/PPO variant of the vendored OpenPI 0.5 BEHAVIOR model wrapper.

Selected by ``actor.model.openpi.task: rl`` in YAML. Adds the chain-collecting
SDE sampler and the VLM-pooled value head needed by PPO:

* :meth:`OpenPiPytorchRLActionModel._predict_train` — chain-collecting SDE
  sampler that emits ``forward_inputs`` ready for PPO.
* :meth:`OpenPiPytorchRLActionModel.default_forward` — PPO recompute of
  logprobs and values from the stored chain.
* :meth:`OpenPiPytorchRLActionModel.setup_wrappers` /
  :meth:`input_transform` / :meth:`output_transform` — the openpi.transforms
  glue (port of ``OpenPi0ForRLActionPrediction.setup_wrappers`` etc.).

Scope (only what BEHAVIOR pi05 PPO exercises): ``noise_method`` ∈
{``flow_ode``, ``flow_sde``}, ``joint_logprob=False``,
``value_after_vlm=True`` with ``value_vlm_mode="mean_token"``. Other
combinations from the original implementation raise
:class:`NotImplementedError`.
"""

from __future__ import annotations

import dataclasses
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Sequence

import numpy as np
import torch
from torch.utils._pytree import tree_map

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.models.embodiment.openpi_pytorch.openpi_action_model import (
    OpenPiPytorchActionModel,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model import model as pi0_model_module
from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0 import Pi0
from rlinf.models.embodiment.openpi_pytorch.utils import rl_sampler


def _to_numpy(x):
    return np.asarray(x.detach().cpu()) if torch.is_tensor(x) else x


@dataclasses.dataclass(frozen=True)
class OpenPiPytorchRLConfig:
    """Static RL knobs read from ``actor.model.openpi`` in YAML."""

    add_value_head: bool = False
    noise_method: str = "flow_ode"
    noise_level: float = 0.0
    joint_logprob: bool = False
    ignore_last: bool = False
    value_after_vlm: bool = False
    value_vlm_mode: str = "mean_token"
    detach_critic_input: bool = False
    train_expert_only: bool = False
    config_name: str = ""


class OpenPiPytorchRLActionModel(OpenPiPytorchActionModel):
    """Eval + PPO variant of :class:`OpenPiPytorchActionModel`."""

    def __init__(
        self,
        pi0_model: Pi0,
        *,
        num_steps: int,
        action_chunk: int,
        action_env_dim: int,
        rl_cfg: OpenPiPytorchRLConfig,
        paligemma_width: int,
    ):
        super().__init__(
            pi0_model,
            processor=None,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
        )
        # RL-only shape knobs: ``action_chunk`` slices the env-action subspace
        # out of the chain logp tensor; ``action_horizon`` / ``model_action_dim``
        # size the SDE chain. The base class / SFT path don't need them.
        self.action_chunk = action_chunk
        self.action_horizon = pi0_model.action_horizon
        self.model_action_dim = pi0_model.action_dim
        self.rl_cfg = rl_cfg

        # openpi.transforms pipeline (installed by ``setup_wrappers``).
        self._input_transform_fn = None
        self._output_transform_fn = None

        if rl_cfg.add_value_head:
            if not rl_cfg.value_after_vlm:
                raise NotImplementedError(
                    "OpenPiPytorchRLActionModel currently only supports the "
                    "value_after_vlm=True value head (BEHAVIOR pi05 config); "
                    "suffix-pooled (value_after_vlm=False) is not wired through."
                )
            # Same MLP shape the original PPO path uses for pi05 (1024, 512, 256).
            self.value_head = ValueHead(
                input_dim=paligemma_width,
                hidden_sizes=(1024, 512, 256),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )
            # Match the wrapped Pi0's parameter dtype so the rollout side (no
            # FSDP, weights already cast to ``cfg.precision``) does the prefix
            # → value head pass in one dtype, and so the FSDP-wrapped actor's
            # MixedPrecisionPolicy(param_dtype=bf16) keeps the value head
            # consistent with the rest of the model.
            model_dtype = next(self.model.parameters()).dtype
            self.value_head.to(model_dtype)

    def set_global_step(self, global_step: int) -> None:
        """Noise-annealing hook — currently a no-op (constant noise_level)."""
        del global_step

    # ------------------------------------------------------------ transforms

    def setup_wrappers(
        self,
        transforms: Sequence,
        output_transforms: Sequence,
    ) -> None:
        """Install the openpi.transforms input/output pipelines.

        ``transforms`` is the list passed to the openpi-side ``compose`` —
        typically ``BehaviorInputs → Normalize(norm_stats) → ModelTransformFactory(model_config)``
        (the last stage carries the auto-downloading PaliGemma tokenizer).
        ``output_transforms`` is the matching reverse pipeline used to turn
        sampled model actions back into env-frame actions.
        """
        from openpi.transforms import compose

        self._input_transform_fn = compose(transforms)
        self._output_transform_fn = compose(output_transforms)

    def _ensure_wrappers(self) -> None:
        if self._input_transform_fn is None or self._output_transform_fn is None:
            raise RuntimeError(
                "OpenPiPytorchRLActionModel.setup_wrappers(...) must be called "
                "after construction (the factory does this); the openpi "
                "transforms pipeline is not yet installed."
            )

    def _repack_env_obs(self, env_obs: dict) -> dict:
        """Map the env's observation dict to the ``observation/*`` keys the openpi pipeline expects."""
        config_name = self.rl_cfg.config_name
        if "behavior" in config_name:
            return {
                "observation/image": env_obs["main_images"],
                "observation/wrist_image": env_obs["wrist_images"],
                "observation/state": env_obs["states"],
                "prompt": env_obs["task_descriptions"],
            }
        raise NotImplementedError(
            f"openpi_pytorch RL repack only knows BEHAVIOR configs today; "
            f"got config_name={config_name!r}."
        )

    def input_transform(self, obs: dict, transpose: bool = False) -> dict:
        """Apply the openpi input pipeline per-sample then recombine into a batched dict.

        Mirrors :meth:`OpenPi0ForRLActionPrediction.input_transform`. Two modes:

        * Rollout (``"prompt"`` key present) — runs the full pipeline including
          prompt tokenization; result has ``image``/``image_mask``/``state``/
          ``tokenized_prompt``/``tokenized_prompt_mask`` keys.
        * Train recompute (no ``"prompt"``; only ``observation/*`` and the cached
          ``tokenized_prompt`` keys present) — re-runs the pipeline using the
          cached tokens rather than re-tokenising every micro-batch.
        """
        self._ensure_wrappers()
        inputs = tree_map(lambda x: x, obs)
        first_process = "prompt" in inputs.keys()
        if first_process:
            inputs.pop("prompt")
        else:
            inputs = {k: inputs[k] for k in inputs.keys() if "/" in k}

        inputs = tree_map(_to_numpy, inputs)
        batch_size = next(v.shape[0] for v in inputs.values() if hasattr(v, "shape"))

        batch_samples = []
        for i in range(batch_size):
            sample = tree_map(lambda x: x[i], inputs)
            if transpose:
                sample = tree_map(
                    lambda x: x.transpose(1, 2, 0)
                    if isinstance(x, np.ndarray) and x.ndim == 3
                    else x,
                    sample,
                )
            if first_process:
                prompts = obs["prompt"]
                if isinstance(prompts, np.ndarray):
                    prompts = prompts.tolist()
                sample["prompt"] = prompts[i]
            else:
                # Pipeline still runs Tokenize, but the cached tokens below
                # overwrite its output — placeholder text is fine.
                sample["prompt"] = "xxxx"
            batch_samples.append(sample)

        with ThreadPoolExecutor(max_workers=min(len(batch_samples), 8)) as ex:
            transformed = list(ex.map(self._input_transform_fn, batch_samples))

        recombined = tree_map(
            lambda *xs: torch.from_numpy(np.asarray(xs).copy()),
            *transformed,
        )
        if not first_process:
            recombined["tokenized_prompt"] = obs["tokenized_prompt"]
            recombined["tokenized_prompt_mask"] = obs["tokenized_prompt_mask"]
        return recombined

    def output_transform(self, outputs: dict) -> dict:
        """Apply the openpi output pipeline per-sample then recombine."""
        self._ensure_wrappers()
        batch_size = outputs["actions"].shape[0]
        transformed = []
        for i in range(batch_size):
            sample = tree_map(
                lambda x: _to_numpy(x[i]) if torch.is_tensor(x) else x[i],
                outputs,
            )
            sample = self._output_transform_fn(sample)
            transformed.append(sample)
        recombined = tree_map(
            lambda *xs: torch.from_numpy(np.asarray(xs).copy()),
            *transformed,
        )
        recombined["actions"] = recombined["actions"][:, : self.action_chunk]
        return recombined

    # ------------------------------------------------------------------ rollout

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        compute_values: bool = False,
        *,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample env actions, in eval (Euler ODE) or train (SDE chain) mode."""
        del kwargs
        self._ensure_wrappers()
        repacked = self._repack_env_obs(env_obs)
        processed = self.input_transform(repacked, transpose=False)
        observation = Observation.from_dict(processed).to_dict()  # fp32 image conversion
        # Move to device
        observation = self._observation_dict_to_device(processed)

        if mode == "eval":
            return self._predict_eval_rl(observation, noise=noise, rng=rng)
        if mode != "train":
            raise ValueError(f"Unknown predict mode: {mode!r}")
        return self._predict_train(
            observation, noise=noise, rng=rng, compute_values=compute_values
        )

    def _observation_dict_to_device(self, processed: dict) -> Observation:
        """Convert a per-key dict (from :meth:`input_transform`) into a device-resident :class:`Observation`."""
        device = self.device
        obs = Observation.from_dict(processed)

        def _move(x):
            return x.to(device) if isinstance(x, torch.Tensor) else x

        return Observation(
            images={k: _move(v) for k, v in obs.images.items()},
            image_masks={k: _move(v) for k, v in obs.image_masks.items()},
            state=_move(obs.state),
            tokenized_prompt=_move(obs.tokenized_prompt),
            tokenized_prompt_mask=_move(obs.tokenized_prompt_mask),
            token_ar_mask=_move(obs.token_ar_mask),
            token_loss_mask=_move(obs.token_loss_mask),
            pcd_xyz=_move(obs.pcd_xyz),
        )

    def _predict_eval_rl(
        self,
        observation: Observation,
        *,
        noise: torch.Tensor | None,
        rng: torch.Generator | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        model_actions = self.model.sample_actions(
            observation, num_steps=self.num_steps, noise=noise, rng=rng
        )
        env_outputs = self.output_transform(
            {"actions": model_actions, "state": observation.state}
        )
        actions = env_outputs["actions"].to(self.device)
        B = actions.shape[0]
        return actions, {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": {
                "action": actions.reshape(B, -1).contiguous(),
                "model_action": model_actions.reshape(B, -1).contiguous(),
            },
        }

    def _predict_train(
        self,
        observation: Observation,
        *,
        noise: torch.Tensor | None,
        rng: torch.Generator | None,
        compute_values: bool,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del compute_values  # always True for PPO; value head presence is the real gate
        rl_cfg = self.rl_cfg
        # joint_logprob=True / NFT collection are not wired through — surface
        # the configuration mismatch loudly so a misconfigured run does not
        # silently produce zero-logprob trajectories.
        if rl_cfg.joint_logprob:
            raise NotImplementedError(
                "openpi_pytorch RL port supports joint_logprob=False only "
                "(single random denoise step). Set actor.model.openpi.joint_logprob=False."
            )
        if not rl_cfg.add_value_head:
            raise RuntimeError(
                "predict_action_batch(mode='train') requires actor.model.add_value_head=True "
                "and openpi.value_after_vlm=True; check the experiment YAML."
            )

        device = self.device
        # Mirror Pi0.sample_actions' invariant: every prefix / suffix pass sees
        # the preprocessed observation (image resize/pad + default masks).
        observation = pi0_model_module.preprocess_observation(observation, train=False)
        B = observation.state.shape[0]
        num_steps = self.num_steps

        if noise is None:
            noise = torch.randn(
                B,
                self.action_horizon,
                self.model_action_dim,
                device=device,
                dtype=torch.float32,
                generator=rng,
            )

        prefix_out, prefix_mask, kv_cache = self.model.build_prefix_cache(observation)

        # VLM-pooled value (BEHAVIOR pi05): one value per sample, reused as the
        # rollout ``prev_values`` independent of the chosen denoise index.
        vlm_value = rl_sampler.value_from_prefix(
            self.value_head, prefix_out, prefix_mask, mode=rl_cfg.value_vlm_mode
        )

        # Single stochastic denoise step picked uniformly; the remaining steps
        # are deterministic flow_ode. ``denoise_inds[:, 0]`` is the read site
        # in default_forward (and matches the original RL contract where every
        # column of denoise_inds carries the same chosen index).
        hi = num_steps - 2 if rl_cfg.ignore_last else num_steps - 1
        chosen = random.randint(0, hi)
        denoise_inds = torch.full(
            (B, num_steps), chosen, device=device, dtype=torch.long
        )
        idx_step = torch.empty(B, device=device, dtype=torch.long)
        timesteps = rl_sampler.get_timesteps(num_steps, device)

        chains = [noise]
        log_probs: list[torch.Tensor] = []
        x_t = noise

        for idx in range(num_steps):
            method = rl_cfg.noise_method if idx == chosen else "flow_ode"
            t_val = float(timesteps[idx].item())
            t_tensor = torch.full(
                (B,), t_val, device=device, dtype=torch.float32
            )
            suffix_act = self.model.run_suffix(
                observation, x_t, t_tensor, kv_cache, prefix_mask
            )
            v_t = self.model.velocity_from_suffix(suffix_act).to(torch.float32)
            idx_step.fill_(idx)
            x_t_mean, x_t_std = rl_sampler.sample_mean_var(
                x_t.to(torch.float32),
                v_t,
                idx_step,
                noise_method=method,
                noise_level=rl_cfg.noise_level,
                num_steps=num_steps,
            )
            step_noise = torch.randn(
                x_t.shape, device=device, dtype=torch.float32, generator=rng
            )
            x_t = x_t_mean + step_noise * x_t_std
            log_probs.append(rl_sampler.gaussian_logprob(x_t, x_t_mean, x_t_std))
            chains.append(x_t)

        x_0 = x_t
        chains_tensor = torch.stack(chains, dim=1).contiguous()  # [B, N+1, H, A]
        log_probs_stacked = torch.stack(log_probs, dim=1)  # [B, N, H, A]
        log_probs_picked = log_probs_stacked[
            torch.arange(B, device=device), denoise_inds[:, 0]
        ][:, : self.action_chunk, : self.action_env_dim]
        log_probs_picked = log_probs_picked.float().contiguous()

        prev_values = vlm_value[:, None].float().contiguous()  # [B, 1]

        env_outputs = self.output_transform(
            {"actions": x_0, "state": observation.state}
        )
        actions = env_outputs["actions"].to(device)

        forward_inputs: dict[str, torch.Tensor] = {
            "chains": chains_tensor,
            "denoise_inds": denoise_inds,
            "obs_state": observation.state.contiguous(),
            "tokenized_prompt": observation.tokenized_prompt.contiguous(),
            "tokenized_prompt_mask": observation.tokenized_prompt_mask.contiguous(),
            "action": actions.reshape(B, -1).contiguous(),
            "model_action": x_0.reshape(B, -1).contiguous(),
        }
        # Flat keys: the rollout layer's `_split_rollout_result` uses a flat
        # torch.split over forward_inputs values, so nested image / mask dicts
        # are unsupported. Encode them with prefixed scalar keys here and decode
        # symmetrically in default_forward.
        for k, v in observation.images.items():
            forward_inputs[f"obs_image__{k}"] = v.contiguous()
        for k, v in observation.image_masks.items():
            forward_inputs[f"obs_image_mask__{k}"] = v.contiguous()

        return actions, {
            "prev_logprobs": log_probs_picked,
            "prev_values": prev_values,
            "forward_inputs": forward_inputs,
        }

    # --------------------------------------------------------------- training

    def forward(self, forward_type: ForwardType = ForwardType.DEFAULT, **kwargs):
        """Dispatch — RL variant only supports :attr:`ForwardType.DEFAULT`."""
        if forward_type != ForwardType.DEFAULT:
            raise NotImplementedError(
                f"{type(self).__name__} only supports ForwardType.DEFAULT; "
                f"got forward_type={forward_type!r}. "
                "Use the SFT subclass (actor.model.openpi.task='sft') for SFT."
            )
        return self.default_forward(**kwargs)

    def default_forward(self, forward_inputs: dict[str, torch.Tensor], **kwargs):
        """PPO-time recompute of logprobs and values from the stored chain."""
        rl_cfg = self.rl_cfg
        if rl_cfg.joint_logprob:
            raise NotImplementedError(
                "openpi_pytorch RL port supports joint_logprob=False only."
            )

        compute_values = kwargs.get("compute_values", True)

        chains = forward_inputs["chains"]
        denoise_inds = forward_inputs["denoise_inds"]
        B = chains.shape[0]
        device = chains.device

        images: dict[str, torch.Tensor] = {}
        image_masks: dict[str, torch.Tensor] = {}
        for key, value in forward_inputs.items():
            if key.startswith("obs_image__"):
                images[key[len("obs_image__") :]] = value
            elif key.startswith("obs_image_mask__"):
                image_masks[key[len("obs_image_mask__") :]] = value
        observation = Observation(
            images=images,
            image_masks=image_masks,
            state=forward_inputs["obs_state"],
            tokenized_prompt=forward_inputs["tokenized_prompt"],
            tokenized_prompt_mask=forward_inputs["tokenized_prompt_mask"],
        )

        # Grad-enabled prefix pass — the VLM value head reads ``prefix_out``
        # so gradients must flow through paligemma here.
        if rl_cfg.train_expert_only:
            with torch.no_grad():
                prefix_out, prefix_mask, kv_cache = self.model.build_prefix_cache(observation)
        else:
            prefix_out, prefix_mask, kv_cache = self.model.build_prefix_cache(observation)

        idx0 = denoise_inds[:, 0].to(torch.long)
        arange_B = torch.arange(B, device=device)
        chains_pre = chains[arange_B, idx0]            # x_t   at the chosen step
        chains_next = chains[arange_B, idx0 + 1]       # x_{t-dt} actually drawn at rollout

        timesteps = rl_sampler.get_timesteps(self.num_steps, device)
        t_input = timesteps[idx0].to(torch.float32)

        suffix_act = self.model.run_suffix(
            observation, chains_pre, t_input, kv_cache, prefix_mask
        )
        v_t = self.model.velocity_from_suffix(suffix_act).to(torch.float32)

        x_t_mean, x_t_std = rl_sampler.sample_mean_var(
            chains_pre.to(torch.float32),
            v_t,
            idx0,
            noise_method=rl_cfg.noise_method,
            noise_level=rl_cfg.noise_level,
            num_steps=self.num_steps,
        )
        log_probs = rl_sampler.gaussian_logprob(
            chains_next.to(torch.float32), x_t_mean, x_t_std
        )
        log_probs = log_probs[
            :, : self.action_chunk, : self.action_env_dim
        ].float().contiguous()

        if compute_values and rl_cfg.add_value_head and rl_cfg.value_after_vlm:
            values = rl_sampler.value_from_prefix(
                self.value_head, prefix_out, prefix_mask, mode=rl_cfg.value_vlm_mode
            )
        else:
            values = torch.zeros(B, device=device, dtype=torch.float32)

        entropy = torch.zeros((B, 1), device=device, dtype=torch.float32)
        return {
            "logprobs": log_probs,
            "values": values.float(),
            "entropy": entropy,
        }

    def freeze_vlm(self) -> int:
        """Freeze the PaliGemma VLM (vision + expert-0 of the LLM) for PPO.
        Returns the count of newly-frozen parameter tensors (for logging).
        """
        frozen = 0
        # SigLIP vision encoder — entirely belongs to the PaliGemma side.
        for p in self.model.img.parameters():
            if p.requires_grad:
                p.requires_grad = False
                frozen += 1
        # Multi-expert gemma: expert index 0 = paligemma_2b LLM, expert 1 = action expert.
        llm = self.model.llm
        # Shared token embedder feeds the paligemma side of the LLM only.
        for p in llm.embedder.parameters():
            if p.requires_grad:
                p.requires_grad = False
                frozen += 1
        # Per-Block per-expert submodules: ModuleLists indexed by expert.
        for block in llm.layers:
            for sub in (
                block.pre_attention_norms[0],
                block.pre_ffw_norms[0],
                block.mlps[0],
            ):
                for p in sub.parameters():
                    if p.requires_grad:
                        p.requires_grad = False
                        frozen += 1
            attn = block.attn
            for proj_list in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
                proj = proj_list[0]
                if proj is None:
                    continue
                for p in proj.parameters():
                    if p.requires_grad:
                        p.requires_grad = False
                        frozen += 1
        # Per-expert final norms.
        if llm.final_norms[0] is not None:
            for p in llm.final_norms[0].parameters():
                if p.requires_grad:
                    p.requires_grad = False
                    frozen += 1
        return frozen
