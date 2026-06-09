# Copyright (c) 2025, RLinf contributors.
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

"""Reproducibility-only pinned-input loader for BEHAVIOR SFT.

This is NOT a normal training path. It is gated by ``data.pinned_inputs_npz`` +
``data.pinned_noise_time_npz`` and exists solely to validate that the actual
FSDP SFT stack, fed the reference's exact first-N inputs, reproduces the
reference first-N loss curve.

It replays a reference rank-0-fanout first-N global-batch sequence dumped to two
``.npz`` files:

* the batches file stores, for every global step ``s`` and fanout chunk ``c``,
  the loader output at flat index ``s * world_size + c`` (so each per-rank micro
  batch is one ``(micro, ...)`` slice);
* the noise/time file stores the SHARED flow-matching ``noise``/``time`` for the
  full global batch of each step, shape ``(n_steps, world_size * micro, ...)``.

Each distributed rank ``r`` consumes the chunk the reference fanout assigned to
position ``r`` — flat index ``s * world_size + r`` (i.e. ``arr[r::world_size]``)
— and the matching noise/time slice ``[r * micro : (r + 1) * micro]``. The
per-step batch carries those ``noise``/``time`` tensors so the SFT forward uses
them verbatim instead of sampling. With FSDP's mean gradient reduction and the
rank-averaged loss logging, this reproduces the reference's
``world_size``-chunk gradient-accumulated step exactly.
"""

from __future__ import annotations

import numpy as np
import torch

from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation

from .behavior_sft_data_loader import BehaviorSftDataConfig

_IMG = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _rank_slice_batches(npz_path: str, rank: int, world_size: int) -> dict:
    """Keep only this rank's ``arr[rank::world_size]`` slice of each array.

    The batches file holds the full global sequence (``n_steps * world_size``
    micro batches). Reading one member at a time and retaining only the rank's
    stride avoids holding ``world_size`` copies of the (multi-GB) image arrays.
    """
    out = {}
    with np.load(npz_path) as f:
        for key in f.files:
            out[key] = np.ascontiguousarray(f[key][rank::world_size])
    return out


class PinnedBehaviorSftDataLoader:
    """Yield this rank's pinned ``{observation, actions, noise, time}`` per step."""

    def __init__(
        self,
        batches: dict,
        noise: np.ndarray,
        time: np.ndarray,
        data_config: BehaviorSftDataConfig,
    ):
        self._batches = batches  # each value: (n_steps, micro, ...)
        self._noise = noise  # (n_steps, micro, action_horizon, action_dim)
        self._time = time  # (n_steps, micro)
        self._data_config = data_config
        self._n_steps = int(batches["actions"].shape[0])
        if not (self._noise.shape[0] == self._time.shape[0] == self._n_steps):
            raise ValueError(
                "Pinned noise/time step count must match the batch step count; "
                f"got noise {self._noise.shape[0]}, time {self._time.shape[0]}, "
                f"batches {self._n_steps}."
            )

    def data_config(self) -> BehaviorSftDataConfig:
        return self._data_config

    def __len__(self) -> int:
        return self._n_steps

    def __iter__(self):
        for step in range(self._n_steps):
            yield self._build_step(step)

    def _build_step(self, step: int) -> dict:
        b = self._batches
        obs: dict = {
            "image": {k: torch.from_numpy(b[f"image__{k}"][step]).float() for k in _IMG}
        }
        if f"image_mask__{_IMG[0]}" in b:
            obs["image_mask"] = {
                k: torch.from_numpy(b[f"image_mask__{k}"][step]) for k in _IMG
            }
        obs["state"] = torch.from_numpy(b["state"][step]).float()
        obs["tokenized_prompt"] = torch.from_numpy(b["tokenized_prompt"][step]).long()
        if "tokenized_prompt_mask" in b:
            obs["tokenized_prompt_mask"] = torch.from_numpy(
                b["tokenized_prompt_mask"][step]
            ).bool()
        return {
            "observation": Observation.from_dict(obs),
            "actions": torch.from_numpy(b["actions"][step]).float(),
            "noise": torch.from_numpy(self._noise[step]).float(),
            "time": torch.from_numpy(self._time[step]).float(),
        }


def build_pinned_behavior_sft_dataloader(cfg, world_size: int, rank: int):
    """Build the reproducibility-only pinned BEHAVIOR SFT loader for one rank.

    Returns ``(loader, data_config)`` to mirror
    :func:`build_behavior_sft_dataloader`'s contract.
    """
    data_cfg = cfg.data
    batches_npz = data_cfg.get("pinned_inputs_npz", None)
    noise_time_npz = data_cfg.get("pinned_noise_time_npz", None)
    if not batches_npz or not noise_time_npz:
        raise ValueError(
            "The pinned BEHAVIOR SFT loader requires data.pinned_inputs_npz and "
            "data.pinned_noise_time_npz."
        )
    micro = int(cfg.actor.micro_batch_size)

    batches = _rank_slice_batches(str(batches_npz), rank, world_size)

    lo = rank * micro
    with np.load(str(noise_time_npz)) as f:
        global_batch = int(f["noise"].shape[1])
        if global_batch != world_size * micro:
            raise ValueError(
                f"Pinned noise global batch {global_batch} != world_size*micro "
                f"({world_size}*{micro}); the dump topology does not match this run."
            )
        noise = np.ascontiguousarray(f["noise"][:, lo : lo + micro])
        time = np.ascontiguousarray(f["time"][:, lo : lo + micro])

    action_horizon = int(batches["actions"].shape[-2])
    action_dim = int(batches["actions"].shape[-1])
    tasks = data_cfg.get("tasks", None)
    data_config = BehaviorSftDataConfig(
        repo_id=str(tasks[0]) if tasks else "pinned",
        action_dim=action_dim,
        action_horizon=action_horizon,
        max_token_len=200,
        norm_stats={},
    )
    loader = PinnedBehaviorSftDataLoader(batches, noise, time, data_config)
    return loader, loader.data_config()
