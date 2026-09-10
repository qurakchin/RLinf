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

"""Pure configuration tests for the scheduler's dual-YAM resource."""

from __future__ import annotations

import copy

import pytest

from rlinf.robotics.discovery import RobotDiscovery
from rlinf.robotics.robots.dual_yam import (
    DualYamConfig,
    DualYamDiscovery,
    DualYamRobot,
    YamCameraConfig,
    YamDeviceConfig,
)
from rlinf.scheduler.hardware import Hardware, NodeHardwareConfig


def _device(channel: str) -> dict:
    return {
        "channel": channel,
        "gripper_type": "yam_gripper",
        "ee_mass": 0.258,
        "gripper_limits": [0.0, 1.0],
    }


def _station_values(prefix: str, *, node_rank: int = 0) -> dict:
    return {
        "node_rank": node_rank,
        "left_follower": _device(f"{prefix}_follower_left"),
        "right_follower": _device(f"{prefix}_follower_right"),
        "left_leader": _device(f"{prefix}_leader_left"),
        "right_leader": _device(f"{prefix}_leader_right"),
        "cameras": [
            {
                "name": "top_rgb",
                "serial": f"{prefix}_camera",
                "resolution": [640, 480],
                "fps": 30,
            }
        ],
    }


def test_dual_yam_is_registered_and_nested_mappings_are_converted():
    values = _station_values("station_a")
    values["left_leader"]["gravity_comp_factor"] = [1, 1, 1, 1, 1, 1]
    values["right_leader"]["ee_mass"] = None

    node_hardware = NodeHardwareConfig(type="DualYam", configs=[values])

    registration = RobotDiscovery.registry["DualYam"]
    assert registration.config_cls is DualYamConfig
    assert registration.robot_cls is DualYamRobot
    assert registration.discovery_cls is DualYamDiscovery
    assert NodeHardwareConfig._hardware_config_registry["DualYam"] is DualYamConfig
    matching_policies = [
        policy for policy in Hardware.policy_registry if policy.HW_TYPE == "DualYam"
    ]
    assert matching_policies == [DualYamDiscovery]
    assert len(node_hardware.configs) == 1
    config = node_hardware.configs[0]
    assert isinstance(config, DualYamConfig)
    assert isinstance(config.left_follower, YamDeviceConfig)
    assert isinstance(config.cameras[0], YamCameraConfig)
    assert config.left_leader.gravity_comp_factor == [1.0] * 6
    assert config.right_leader.ee_mass is None


def test_a_station_needs_all_four_devices_and_a_camera():
    """The fields are defaulted, so __post_init__ is what requires them."""
    for field_name in (
        "left_follower",
        "right_follower",
        "left_leader",
        "right_leader",
    ):
        values = _station_values("incomplete")
        del values[field_name]
        with pytest.raises(ValueError, match=f"'{field_name}'.*must name a CAN device"):
            DualYamConfig(**values)

    values = _station_values("incomplete")
    values["cameras"] = []
    with pytest.raises(ValueError, match="at least one camera"):
        DualYamConfig(**values)


def test_enumeration_returns_one_info_per_complete_station_and_filters_node_rank():
    local = DualYamConfig(**_station_values("local", node_rank=0))
    remote = DualYamConfig(**_station_values("remote", node_rank=1))

    resource = DualYamDiscovery.enumerate(0, [local, remote])

    assert resource is not None
    assert resource.type == "DualYam"
    assert resource.count == 1
    assert resource.infos[0].config is local
    assert resource.infos[0].model == "DualYam"
    assert DualYamDiscovery.enumerate(2, [local, remote]) is None


def test_station_rejects_a_can_channel_reused_by_two_devices():
    values = _station_values("station")
    values["right_leader"]["channel"] = values["left_follower"]["channel"]

    with pytest.raises(ValueError, match="four YAM.*CAN channels must be unique"):
        DualYamConfig(**values)


def test_gripper_limits_preserve_a_descending_closed_to_open_calibration():
    device = YamDeviceConfig(
        channel="can0",
        gripper_type="flexible_4310",
        ee_mass=None,
        gripper_limits=(1.2, -0.4),
    )

    assert device.gripper_limits == (1.2, -0.4)


def test_gripper_limits_allow_i2rt_auto_calibration():
    device = YamDeviceConfig(
        channel="can0",
        gripper_type="flexible_4310",
        ee_mass=None,
        gripper_limits=None,
    )

    assert device.gripper_limits is None


def test_gripper_limits_reject_identical_closed_and_open_stops():
    with pytest.raises(ValueError, match="closed and open stops must differ"):
        YamDeviceConfig(
            channel="can0",
            gripper_type="flexible_4310",
            ee_mass=None,
            gripper_limits=(0.5, 0.5),
        )


@pytest.mark.parametrize("conflict", ["can", "camera"])
def test_enumeration_rejects_resources_shared_between_stations(conflict: str):
    first = DualYamConfig(**_station_values("first"))
    second_values = copy.deepcopy(_station_values("second"))
    if conflict == "can":
        second_values["left_follower"]["channel"] = first.left_follower.channel
        message = "YAM CAN channel.*is shared"
    else:
        second_values["cameras"][0]["serial"] = first.cameras[0].serial
        message = "YAM camera serial.*is shared"
    second = DualYamConfig(**second_values)

    with pytest.raises(ValueError, match=message):
        DualYamDiscovery.enumerate(0, [first, second])
