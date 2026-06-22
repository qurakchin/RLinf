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

"""Abstract base for the self-contained PyTorch OpenPI 0.5 BEHAVIOR model.

Holds only what is genuinely shared by every concrete variant: the wrapped
``Pi0`` and eval processor, the deterministic Euler ODE sampler used by all
eval / action-generation paths, and the gradient-checkpointing pass-through
for the FSDP manager. Anything specific to a training stage — SFT loss, RL
chain sampling, value head — lives in the concrete subclasses
(:mod:`rlinf.models.embodiment.openpi_pytorch.sft_action_model` and
:mod:`rlinf.models.embodiment.openpi_pytorch.rl_action_model`).

The factory in :mod:`rlinf.models.embodiment.openpi_pytorch` picks the
concrete subclass off the YAML field ``actor.model.openpi.task``
(``"sft"`` / ``"rl"``). ``actions`` returned by ``predict_action_batch`` have
shape ``[B, action_chunk, action_env_dim]`` (e.g. ``[B, 32, 23]`` for
BEHAVIOR pi05).
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn

from rlinf.data.datasets.openpi_pytorch.eval_processor import EvalProcessor
from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0 import Pi0


class OpenPiPytorchActionModel(nn.Module):
    """Eval-capable wrapper around the vendored ``Pi0`` model.

    Concrete subclasses extend this base with a training-time forward:

    * :class:`OpenPiPytorchSFTActionModel` adds :meth:`forward` /
      :meth:`sft_forward` (flow-matching loss).
    * :class:`OpenPiPytorchRLActionModel` adds train-mode rollouts
      (:meth:`predict_action_batch` ``mode="train"``) and a PPO recompute
      :meth:`forward` / :meth:`default_forward`.
    """

    def __init__(
        self,
        pi0_model: Pi0,
        processor: EvalProcessor | None,
        *,
        num_steps: int,
        action_env_dim: int,
    ):
        super().__init__()
        self.model = pi0_model
        self.processor = processor
        self.num_steps = num_steps
        self.action_env_dim = action_env_dim

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

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
        """Sample env actions via the deterministic Euler ODE sampler.

        Only ``mode="eval"`` is supported at the base level — that path is
        shared by every variant. ``mode="train"`` is implemented in the
        RL subclass which overrides this method to add the chain-collecting
        SDE sampler. Calling with ``mode="train"`` on the base raises
        :class:`NotImplementedError` so an SFT-only model loudly refuses to
        be used for on-policy rollouts.
        """
        del compute_values, kwargs  # accepted for call-site parity; eval ignores them
        if mode != "eval":
            raise NotImplementedError(
                f"{type(self).__name__} only supports predict_action_batch(mode='eval'); "
                "use the RL subclass (actor.model.openpi.task='rl') for train rollouts."
            )
        if self.processor is None:
            raise RuntimeError(
                "predict_action_batch requires an eval processor; "
                "the current model was built without one."
            )
        observation = self.processor.build_observation(env_obs, self.device)
        return self._predict_eval(observation, noise=noise, rng=rng)

    def _predict_eval(
        self,
        observation: Observation,
        *,
        noise: torch.Tensor | None,
        rng: torch.Generator | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        model_actions = self.model.sample_actions(
            observation, num_steps=self.num_steps, noise=noise, rng=rng
        )
        actions = self.processor.postprocess_actions(model_actions).to(self.device)
        B = actions.shape[0]
        result = {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": {
                "action": actions.reshape(B, -1).contiguous(),
                "model_action": model_actions.reshape(B, -1).contiguous(),
            },
        }
        return actions, result

    # --- Gradient checkpointing pass-through (used by the FSDP training path) ---
    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict | None = None, **kwargs
    ) -> None:
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self, **kwargs) -> None:
        self.model.gradient_checkpointing_disable()
