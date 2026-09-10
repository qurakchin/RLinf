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

"""Contracts shared by dual-YAM SFT and deployment transforms."""

import numpy as np
import pytest

pytest.importorskip("openpi.transforms")

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.policies.yam_policy import YamInputs, YamOutputs


def _sample_image(value: int) -> np.ndarray:
    return np.full((3, 8, 10), value, dtype=np.uint8)


def test_yam_inputs_accept_split_training_views():
    transformed = YamInputs()(
        {
            "observation/image": _sample_image(0),
            "observation/extra_view_image-0": _sample_image(1),
            "observation/extra_view_image-1": _sample_image(2),
            "observation/state": np.arange(14, dtype=np.float32),
            "actions": np.zeros((50, 14), dtype=np.float32),
            "prompt": b"pick block",
        }
    )

    assert transformed["image"]["base_0_rgb"].shape == (8, 10, 3)
    assert transformed["image"]["left_wrist_0_rgb"][0, 0, 0] == 1
    assert transformed["image"]["right_wrist_0_rgb"][0, 0, 0] == 2
    assert transformed["state"].shape == (14,)
    assert transformed["actions"].shape == (50, 14)
    assert transformed["prompt"] == "pick block"


def test_yam_inputs_accept_stacked_deployment_views():
    transformed = YamInputs()(
        {
            "observation/image": _sample_image(0),
            "observation/extra_view_image": np.stack(
                [_sample_image(3), _sample_image(4)]
            ),
            "observation/state": np.arange(14, dtype=np.float32),
            "prompt": "pick block",
        }
    )

    assert transformed["image"]["left_wrist_0_rgb"][0, 0, 0] == 3
    assert transformed["image"]["right_wrist_0_rgb"][0, 0, 0] == 4


def test_yam_config_matches_official_pi05_shape_and_output_contract():
    config = get_openpi_config("pi05_yam_joint")

    assert config.model.pi05 is True
    assert config.model.action_horizon == 50
    assert config.model.action_dim == 32
    assert config.model.discrete_state_input is True
    assert YamOutputs()({"actions": np.zeros((50, 32))})["actions"].shape == (
        50,
        14,
    )
