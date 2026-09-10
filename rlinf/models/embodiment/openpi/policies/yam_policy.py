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

"""Policy transforms for RLinf dual-YAM joint-space data.

The state and action layout is ``[left: 6 joints + gripper, right: 6 joints +
gripper]``. Training receives three split LeRobot v2.1 image features, while
deployment receives the two non-primary views stacked by ``RealWorldEnv``.
"""

import dataclasses

import einops
import numpy as np
from openpi import transforms

YAM_ACTION_DIM = 14


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _extract_side_views(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return the left and right views for training or deployment input."""
    stacked = data.get("observation/extra_view_image")
    if stacked is not None:
        views = np.asarray(stacked)
        if views.shape[0] != 2:
            raise ValueError(
                "YAM deployment expects two extra views ordered [left, right], "
                f"got shape {views.shape}."
            )
        return _parse_image(views[0]), _parse_image(views[1])

    return (
        _parse_image(data["observation/extra_view_image-0"]),
        _parse_image(data["observation/extra_view_image-1"]),
    )


@dataclasses.dataclass(frozen=True)
class YamInputs(transforms.DataTransformFn):
    """Map RLinf YAM observations onto the three Pi0/Pi0.5 image slots."""

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"])
        if state.shape[-1] != YAM_ACTION_DIM:
            raise ValueError(
                f"YAM state must have {YAM_ACTION_DIM} values, got {state.shape}."
            )

        top = _parse_image(data["observation/image"])
        left, right = _extract_side_views(data)
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": top,
                "left_wrist_0_rgb": left,
                "right_wrist_0_rgb": right,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1] != YAM_ACTION_DIM:
                raise ValueError(
                    f"YAM actions must have {YAM_ACTION_DIM} values, "
                    f"got {actions.shape}."
                )
            inputs["actions"] = actions

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class YamOutputs(transforms.DataTransformFn):
    """Strip model padding and return a 14-D absolute YAM action chunk."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :YAM_ACTION_DIM]}
