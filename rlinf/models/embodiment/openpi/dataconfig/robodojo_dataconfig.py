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
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import robodojo_policy


@dataclasses.dataclass(frozen=True)
class LeRobotRoboDojoDataConfig(DataConfigFactory):
    """Data configuration for the RoboDojo dual-arm ARX-X5 joint dataset.

    The 14-D action/state layout is ``[left_arm(6), right_arm(6),
    left_gripper(1), right_gripper(1)]``.
    """

    # RoboDojo raw joint targets are absolute, so they are converted to the
    # delta actions the Pi0 model was pretrained on. Grippers stay absolute.
    extra_delta_transform: bool = True

    # Maps dataset keys onto the inference-time keys produced by the RPent
    # RoboDojo obs encoder. Applied to dataset samples only, never at inference.
    repack_transforms: _transforms.Group = dataclasses.field(
        default_factory=lambda: _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.cam_high",
                        "observation/wrist_image": (
                            "observation.images.cam_left_wrist"
                        ),
                        "observation/extra_view_image": (
                            "observation.images.cam_right_wrist"
                        ),
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
    )

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[robodojo_policy.RoboDojoInputs(model_type=model_config.model_type)],
            outputs=[robodojo_policy.RoboDojoOutputs()],
        )

        if self.extra_delta_transform:
            # Joint channels are delta, gripper channels are absolute.
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
