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
import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

# RoboDojo dual-arm ARX-X5 joint layout: [left_arm(6), right_arm(6), left_gripper(1),
# right_gripper(1)]. The gripper channels are the ``ee_joint_state`` values, where the
# simulator uses 1.0 for open and 0.0 for closed.
ROBODOJO_ACTION_DIM = 14


def make_robodojo_example() -> dict:
    """Creates a random input example for the RoboDojo policy."""
    return {
        "observation/state": np.random.rand(ROBODOJO_ACTION_DIM),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/extra_view_image": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RoboDojoInputs(transforms.DataTransformFn):
    """Convert RoboDojo observations into OpenPI model inputs.

    RoboDojo exposes three views: the head camera (``observation/image``), the
    left wrist (``observation/wrist_image``) and the right wrist
    (``observation/extra_view_image``). They map one-to-one onto the three
    OpenPI image slots, so no view is masked out.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        left_wrist_image = _parse_image(data["observation/wrist_image"])
        right_wrist_image = data.get("observation/extra_view_image")
        if right_wrist_image is None:
            right_wrist_image = np.zeros_like(base_image)
            right_wrist_mask = np.False_
        else:
            right_wrist_image = _parse_image(right_wrist_image)
            right_wrist_mask = np.True_

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": right_wrist_mask,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class RoboDojoOutputs(transforms.DataTransformFn):
    """Convert OpenPI outputs to the RoboDojo 14-D joint action layout."""

    def __call__(self, data: dict) -> dict:
        # Dropping the padding back to the model action dimension.
        return {"actions": np.asarray(data["actions"][:, :ROBODOJO_ACTION_DIM])}
