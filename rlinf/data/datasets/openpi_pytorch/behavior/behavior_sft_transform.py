# Copyright 2026 The RLinf Authors.
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

"""Per-sample BEHAVIOR-1K SFT transform for the PyTorch pi05 path.

This module implements the old (installed-``openpi``) preprocessing pipeline as a
single self-contained per-sample transform, reusing only the vendored
``openpi_pytorch`` primitives. Given one raw LeRobot frame, it produces the model
input dict consumed by :class:`Observation.from_dict` plus padded actions.

The transform reproduces the old chain (repack -> ``BehaviorInputs`` ->
``Normalize`` -> ``ResizeImages`` -> ``TokenizePrompt`` -> ``PadStatesAndActions``)
for the pi05 / direct-task path:

1. Repack the LeRobot keys onto the ``observation/*`` names ``BehaviorInputs``
   reads, and lift the task text into ``prompt``.
2. ``BehaviorInputs`` extracts the 23-dim R1Pro state, maps the three camera
   views to ``base_0_rgb`` / ``left_wrist_0_rgb`` / ``right_wrist_0_rgb``, and
   sets every image mask to ``True``.
3. Resize each camera image to 224x224 with aspect-preserving zero padding
   (uint8 in, uint8 out).
4. Quantile-normalize the 23-dim state to ``[-1, 1]`` (before padding).
5. Quantile-normalize the actions to ``[-1, 1]`` (before padding).
6. Tokenize with the pi05 discrete-state prompt
   (``"Task: {task}, State: {discretized_state};\\nAction: "``) to fixed-length
   token ids and a token mask.
7. Zero-pad the state ``23 -> action_dim`` and the actions
   ``(action_horizon, 23) -> (action_horizon, action_dim)`` (after normalize).

Images stay ``uint8`` through resize; the final ``uint8 -> float[-1, 1]``
conversion is performed by :meth:`Observation.from_dict`, matching the old path.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from rlinf.models.embodiment.openpi_pytorch.utils.normalize import (
    NormStats,
    normalize_quantile,
)
from rlinf.models.embodiment.openpi_pytorch.utils.tokenizer import (
    PaligemmaTokenizer,
)
from rlinf.models.embodiment.openpi_pytorch.policies.behavior_policy import (
    BehaviorInputs,
)
from rlinf.models.embodiment.openpi_pytorch.utils.image_tools import resize_with_pad

# Camera views resolved by `BehaviorInputs` for the pi05 model.
_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_IMAGE_SIZE = 224

# Repack mapping: BehaviorInputs key -> raw LeRobot frame key. Mirrors the old
# LeRobotB1KDataConfig RepackTransform (with the BehaviorInputs key names).
_REPACK_KEYS = {
    "observation/image": "observation.images.rgb.head",
    "observation/left_wrist_image": "observation.images.rgb.left_wrist",
    "observation/right_wrist_image": "observation.images.rgb.right_wrist",
    "observation/state": "observation.state",
}


def _pad_to_dim(x: np.ndarray, target_dim: int, value: float = 0.0) -> np.ndarray:
    """Zero-pad the last axis of ``x`` up to ``target_dim`` (no-op if already >=)."""
    current_dim = x.shape[-1]
    if current_dim >= target_dim:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (0, target_dim - current_dim)
    return np.pad(x, pad_width, constant_values=value)


def _repack(frame: dict) -> dict:
    """Map raw LeRobot keys onto the names ``BehaviorInputs`` expects.

    Images arrive as ``(C, H, W)`` float tensors from the streaming dataset;
    ``BehaviorInputs`` (via its ``_parse_image``) handles the channel order and
    the float-to-uint8 conversion, so they are passed through as numpy arrays.
    """
    data: dict = {}
    for dst, src in _REPACK_KEYS.items():
        data[dst] = np.asarray(frame[src])

    actions = frame.get("action")
    if actions is not None:
        data["actions"] = np.asarray(actions)

    prompt = frame.get("prompt", frame.get("task"))
    if prompt is None:
        raise ValueError(
            "BEHAVIOR SFT frame is missing both 'prompt' and 'task'; the streaming "
            "dataset must set the per-frame task text."
        )
    if not isinstance(prompt, str):
        prompt = prompt.item() if hasattr(prompt, "item") else str(prompt)
    data["prompt"] = prompt
    return data


@dataclasses.dataclass
class BehaviorSftTransform:
    """Map a raw BEHAVIOR LeRobot frame to pi05 model inputs + padded actions.

    Args:
        norm_stats: Quantile normalization statistics keyed by ``"state"`` and
            ``"actions"`` (as loaded from the checkpoint ``norm_stats.json``).
        action_dim: Model action dimension to pad the state and actions to
            (32 for the BEHAVIOR pi05 model).
        max_token_len: Maximum tokenized-prompt length (200 for pi05 BEHAVIOR).
        image_size: Target square image resolution (224).
        tokenizer: Optional pre-built tokenizer. A new
            :class:`PaligemmaTokenizer` is created lazily per worker when ``None``
            so the (non-picklable) SentencePiece processor is not shared across
            ``spawn`` workers.
    """

    norm_stats: dict[str, NormStats]
    action_dim: int = 32
    max_token_len: int = 200
    image_size: int = _IMAGE_SIZE
    tokenizer: PaligemmaTokenizer | None = None

    def __post_init__(self):
        self._behavior_inputs = BehaviorInputs(
            extract_state_from_proprio=True,
            use_all_wrist_images=True,
        )

    def _get_tokenizer(self) -> PaligemmaTokenizer:
        if self.tokenizer is None:
            self.tokenizer = PaligemmaTokenizer(max_len=self.max_token_len)
        return self.tokenizer

    def __call__(self, frame: dict) -> dict:
        """Transform a single raw LeRobot frame into the model-input dict."""
        # 1) Repack LeRobot keys -> BehaviorInputs keys (+ prompt).
        repacked = _repack(frame)

        # 2) BehaviorInputs: 23-dim state extraction, image-key mapping, masks.
        inputs = self._behavior_inputs(repacked)

        # 3) Resize each camera image to image_size x image_size (uint8 in/out).
        images = {
            key: resize_with_pad(
                np.asarray(inputs["image"][key]), self.image_size, self.image_size
            )
            for key in _IMAGE_KEYS
        }

        # 4) Quantile-normalize the (still 23-dim) state to [-1, 1] BEFORE padding.
        state = np.asarray(inputs["state"], dtype=np.float32)
        state = normalize_quantile(state, self.norm_stats["state"]).astype(np.float32)

        # 5) Quantile-normalize the (still 23-dim) actions to [-1, 1] BEFORE padding.
        actions = np.asarray(inputs["actions"], dtype=np.float32)
        actions = normalize_quantile(actions, self.norm_stats["actions"]).astype(
            np.float32
        )

        # 6) Tokenize with the pi05 discrete-state prompt on the normalized state.
        tokens, token_masks = self._get_tokenizer().tokenize(inputs["prompt"], state)

        # 7) Zero-pad the state and actions to the model action dimension.
        state = _pad_to_dim(state, self.action_dim).astype(np.float32)
        actions = _pad_to_dim(actions, self.action_dim).astype(np.float32)

        return {
            "image": images,
            "image_mask": {
                key: np.asarray(inputs["image_mask"][key]) for key in _IMAGE_KEYS
            },
            "state": state,
            "actions": actions,
            "tokenized_prompt": np.asarray(tokens),
            "tokenized_prompt_mask": np.asarray(token_masks),
        }


def transform_behavior_sft_item(frame: dict, transform: BehaviorSftTransform) -> dict:
    """Apply ``transform`` to a single raw LeRobot ``frame`` (functional wrapper)."""
    return transform(frame)
