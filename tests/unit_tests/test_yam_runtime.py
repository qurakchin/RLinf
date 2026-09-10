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

"""Safety-boundary tests for the single-writer YAM control runtime."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.envs.real.yam import i2rt_backend
from rlinf.envs.real.yam.config import DualYamJointEnvConfig
from rlinf.envs.real.yam.control_runtime import YamControlRuntime
from rlinf.envs.real.yam.types import YamArmState, YamLeaderState


class _Follower:
    def __init__(self, device, clock, events):
        self.name = device.name
        self.action = np.asarray(device.initial_action, dtype=np.float64).copy()
        self._fail_connect = bool(device.fail_connect)
        self._clock = clock
        self._events = events
        self.connected = False
        self.timestamp_s = None
        self.commands = []
        self.hold_calls = 0
        self.close_calls = 0

    def connect(self):
        self._events.append(("connect", self.name))
        if self._fail_connect:
            raise RuntimeError(f"failed to connect {self.name}")
        self.connected = True

    def read_state(self):
        assert self.connected
        timestamp_s = self._clock() if self.timestamp_s is None else self.timestamp_s
        return YamArmState(self.action[:6], self.action[6], timestamp_s)

    def command(self, target):
        assert self.connected
        target = np.asarray(target, dtype=np.float64).copy()
        self.commands.append(target)
        self.action = target

    def joint_limits(self):
        return np.column_stack([np.full(6, -np.pi), np.full(6, np.pi)])

    def hold(self):
        assert self.connected
        self.hold_calls += 1
        self._events.append(("hold", self.name))

    def assert_healthy(self, max_feedback_age_s):
        del max_feedback_age_s
        assert self.connected

    def close(self):
        self.close_calls += 1
        self.connected = False
        self._events.append(("close", self.name))


class _Leader:
    def __init__(self, device, clock, events):
        self.name = device.name
        self.action = np.asarray(device.initial_action, dtype=np.float64).copy()
        self.buttons = tuple(device.buttons)
        self._joint_limits = np.asarray(device.joint_limits, dtype=np.float64)
        self._clock = clock
        self._events = events
        self.connected = False
        self.close_calls = 0
        self.commands = []
        self.release_calls = 0

    def connect(self):
        self.connected = True
        self._events.append(("connect", self.name))

    def read_state(self):
        assert self.connected
        arm = YamArmState(self.action[:6], self.action[6], self._clock())
        return YamLeaderState(arm=arm, buttons=self.buttons)

    def command_feedback(self, follower_joints):
        del follower_joints
        assert self.connected

    def command(self, joints):
        assert self.connected
        joints = np.asarray(joints, dtype=np.float64).reshape(6).copy()
        self.commands.append(joints)
        self.action[:6] = joints

    def joint_limits(self):
        return self._joint_limits.copy()

    def release_feedback(self):
        self.release_calls += 1
        return None

    def assert_healthy(self, max_feedback_age_s):
        del max_feedback_age_s
        assert self.connected

    def close(self):
        self.close_calls += 1
        self.connected = False
        self._events.append(("close", self.name))


class _Factory:
    def __init__(self, clock):
        self.clock = clock
        self.events = []
        self.followers = []
        self.leaders = []

    def create_follower(self, device):
        self.events.append(("create", device.name))
        backend = _Follower(device, self.clock, self.events)
        self.followers.append(backend)
        return backend

    def create_leader(self, device):
        self.events.append(("create", device.name))
        backend = _Leader(device, self.clock, self.events)
        self.leaders.append(backend)
        return backend


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _device(
    name: str,
    action,
    buttons=(False, False),
    *,
    fail_connect=False,
    joint_limits=None,
):
    if joint_limits is None:
        joint_limits = np.column_stack([np.full(6, -np.pi), np.full(6, np.pi)])
    return SimpleNamespace(
        name=name,
        initial_action=np.asarray(action, dtype=np.float64),
        buttons=buttons,
        fail_connect=fail_connect,
        joint_limits=np.asarray(joint_limits, dtype=np.float64),
    )


def _hardware(left_action=None, right_action=None, *, right_follower_fails=False):
    left_action = np.zeros(7) if left_action is None else left_action
    right_action = np.zeros(7) if right_action is None else right_action
    return SimpleNamespace(
        left_follower=_device("left_follower", left_action),
        right_follower=_device(
            "right_follower", right_action, fail_connect=right_follower_fails
        ),
        left_leader=_device("left_leader", np.zeros(7)),
        right_leader=_device("right_leader", np.zeros(7)),
    )


def _runtime(config=None, hardware=None):
    clock = _Clock()
    factory = _Factory(clock)
    runtime = YamControlRuntime(
        config or DualYamJointEnvConfig(),
        hardware or _hardware(),
        factory,
        clock=clock,
        sleeper=lambda _seconds: None,
    )
    return runtime, factory


def test_runtime_construction_is_disconnected_and_state_order_is_fixed():
    left = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    right = np.array([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 0.25])
    runtime, factory = _runtime(hardware=_hardware(left, right))

    assert factory.events == []
    assert not runtime.followers_connected
    assert not runtime.leaders_connected

    runtime.connect_followers()

    np.testing.assert_allclose(
        runtime.read_state().as_vector(), np.concatenate([left, right])
    )
    assert [backend.hold_calls for backend in factory.followers] == [1, 1]
    assert factory.events.index(("hold", "left_follower")) < factory.events.index(
        ("create", "right_follower")
    )
    assert [backend.name for backend in factory.followers] == [
        "left_follower",
        "right_follower",
    ]


def test_follower_connection_failure_closes_both_created_backends():
    runtime, factory = _runtime(hardware=_hardware(right_follower_fails=True))

    with pytest.raises(RuntimeError, match="failed to connect right_follower"):
        runtime.connect_followers()

    assert not runtime.followers_connected
    assert len(factory.followers) == 2
    assert [backend.close_calls for backend in factory.followers] == [1, 1]


@pytest.mark.parametrize(
    "invalid_action",
    [np.zeros((2, 7), dtype=np.float64), np.zeros(13, dtype=np.float64)],
    ids=["matrix", "wrong_length"],
)
def test_invalid_action_shape_is_rejected_before_any_write(invalid_action):
    runtime, factory = _runtime()
    runtime.connect_followers()

    with pytest.raises(ValueError, match="shape"):
        runtime.command(invalid_action)

    assert factory.followers[0].commands == []
    assert factory.followers[1].commands == []
    assert [backend.hold_calls for backend in factory.followers] == [1, 1]


def test_command_applies_joint_limits_slew_limits_and_gripper_bounds():
    config = DualYamJointEnvConfig(
        max_joint_delta=0.2,
        joint_limit_min=[[-1.0] * 6, [-0.1, -1.0, -1.0, -1.0, -1.0, -1.0]],
        joint_limit_max=[[0.1, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0] * 6],
    )
    runtime, factory = _runtime(config=config)
    runtime.connect_followers()
    requested = np.array(
        [5.0, 5.0, -5.0, -5.0, 0.05, -0.05, 2.0]
        + [-5.0, -5.0, 5.0, 5.0, 0.05, -0.05, -1.0]
    )

    result = runtime.command(requested)

    expected_left = np.array([0.1, 0.2, -0.2, -0.2, 0.05, -0.05, 1.0])
    expected_right = np.array([-0.1, -0.2, 0.2, 0.2, 0.05, -0.05, 0.0])
    assert result.clipped
    assert result.rejection_reason is None
    np.testing.assert_allclose(
        result.accepted, np.concatenate([expected_left, expected_right])
    )
    np.testing.assert_allclose(factory.followers[0].commands, [expected_left])
    np.testing.assert_allclose(factory.followers[1].commands, [expected_right])


def test_command_can_match_legacy_direct_teleop_mapping():
    config = DualYamJointEnvConfig(
        enforce_runtime_joint_limits=False,
        max_joint_delta=0.01,
        joint_limit_min=[[-1.0] * 6, [-1.0] * 6],
        joint_limit_max=[[1.0] * 6, [1.0] * 6],
    )
    left = np.array([1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    right = np.array([-1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    runtime, factory = _runtime(config=config, hardware=_hardware(left, right))
    runtime.connect_followers()
    requested_left = np.array([2.0, 0.2, 0.3, 0.4, 0.5, 0.6, 2.0])
    requested_right = np.array([-2.0, -0.2, -0.3, -0.4, -0.5, -0.6, -1.0])
    requested = np.concatenate([requested_left, requested_right])

    result = runtime.command(requested)

    expected_left = requested_left.copy()
    expected_left[-1] = 1.0
    expected_right = requested_right.copy()
    expected_right[-1] = 0.0
    np.testing.assert_allclose(factory.followers[0].commands, [expected_left])
    np.testing.assert_allclose(factory.followers[1].commands, [expected_right])
    np.testing.assert_allclose(
        result.accepted, np.concatenate([expected_left, expected_right])
    )
    assert result.rejection_reason is None
    assert result.clipped


def test_move_to_uses_smooth_bounded_interpolation_and_holds_at_target():
    config = DualYamJointEnvConfig(
        step_frequency=10.0,
        max_joint_delta=0.2,
    )
    runtime, factory = _runtime(config=config)
    runtime.connect_followers()
    target = np.array([0.2] * 6 + [1.0] + [0.1] * 6 + [0.5])

    measured = runtime.move_to(
        target,
        duration_s=0.2,
        max_joint_delta=0.2,
        tolerance=1e-6,
        timeout_s=1.0,
    )

    assert len(factory.followers[0].commands) == 2
    np.testing.assert_allclose(factory.followers[0].commands[0], target[:7] * 0.5)
    np.testing.assert_allclose(factory.followers[1].commands[0], target[7:] * 0.5)
    np.testing.assert_allclose(factory.followers[0].commands[-1], target[:7])
    np.testing.assert_allclose(factory.followers[1].commands[-1], target[7:])
    np.testing.assert_allclose(measured, target)
    assert [backend.hold_calls for backend in factory.followers] == [2, 2]


def test_move_to_rejects_target_outside_hard_limits_before_writing():
    config = DualYamJointEnvConfig(
        joint_limit_min=[[-1.0] * 6, [-1.0] * 6],
        joint_limit_max=[[1.0] * 6, [1.0] * 6],
    )
    runtime, factory = _runtime(config=config)
    runtime.connect_followers()
    target = np.zeros(14)
    target[0] = 1.1

    with pytest.raises(ValueError, match="outside joint limits"):
        runtime.move_to(
            target,
            duration_s=0.0,
            max_joint_delta=0.1,
            tolerance=0.01,
            timeout_s=1.0,
        )

    assert factory.followers[0].commands == []
    assert factory.followers[1].commands == []


def test_validate_leader_target_checks_sdk_limits_without_writing():
    limits = np.column_stack([np.full(6, -0.5), np.full(6, 0.5)])
    hardware = _hardware()
    hardware.left_leader = _device("left_leader", np.zeros(7), joint_limits=limits)
    hardware.right_leader = _device("right_leader", np.zeros(7), joint_limits=limits)
    runtime, factory = _runtime(hardware=hardware)
    runtime.connect_leaders()
    target = np.array([0.1] * 6 + [1.0] + [0.2] * 6 + [0.0])

    left, right = runtime.validate_leader_target(target)

    np.testing.assert_allclose(left, target[:7])
    np.testing.assert_allclose(right, target[7:])
    assert factory.leaders[0].commands == []
    assert factory.leaders[1].commands == []

    invalid = target.copy()
    invalid[8] = 0.6
    with pytest.raises(ValueError, match="outside joint limits"):
        runtime.validate_leader_target(invalid)
    assert factory.leaders[0].commands == []
    assert factory.leaders[1].commands == []


def test_command_leaders_prevalidates_then_commands_six_joints():
    runtime, factory = _runtime()
    runtime.connect_followers()
    runtime.connect_leaders()
    target = np.array([0.1] * 6 + [0.8] + [-0.2] * 6 + [0.3])

    runtime.command_leaders(target)

    np.testing.assert_allclose(factory.leaders[0].commands, [target[:6]])
    np.testing.assert_allclose(factory.leaders[1].commands, [target[7:13]])
    assert factory.leaders[0].action[6] == 0.0
    assert factory.leaders[1].action[6] == 0.0


def test_command_leaders_error_holds_followers_and_releases_leaders():
    limits = np.column_stack([np.full(6, -0.5), np.full(6, 0.5)])
    hardware = _hardware()
    hardware.left_leader = _device("left_leader", np.zeros(7), joint_limits=limits)
    hardware.right_leader = _device("right_leader", np.zeros(7), joint_limits=limits)
    runtime, factory = _runtime(hardware=hardware)
    runtime.connect_followers()
    runtime.connect_leaders()
    target = np.zeros(14)
    target[0] = 0.6

    with pytest.raises(ValueError, match="outside joint limits"):
        runtime.command_leaders(target)

    assert factory.leaders[0].commands == []
    assert factory.leaders[1].commands == []
    assert [backend.hold_calls for backend in factory.followers] == [2, 2]
    assert [backend.release_calls for backend in factory.leaders] == [1, 1]


def test_i2rt_close_joins_hidden_can_thread_before_closing_socket(monkeypatch):
    events = []

    class _Event:
        def set(self):
            events.append("stop_server")

    class _Thread:
        def __init__(self, name):
            self.name = name
            self.alive = True

        def join(self, timeout=None):
            events.append(("join", self.name, timeout))
            self.alive = False

        def is_alive(self):
            return self.alive

    server = _Thread("server")
    control = _Thread("control")

    class _Robot:
        _stop_event = _Event()
        _server_thread = server
        motor_chain = SimpleNamespace(running=True, _control_thread=control)

        def close(self):
            assert not server.is_alive()
            assert not control.is_alive()
            events.append("close_socket")

    monkeypatch.setattr(
        i2rt_backend,
        "_build_yam",
        lambda device_config, *, zero_gravity_mode: _Robot(),
    )
    backend = i2rt_backend.I2RTYamFollower(SimpleNamespace(channel="can_left"))
    backend.connect()

    backend.close()

    assert events == [
        "stop_server",
        ("join", "server", 2.0),
        ("join", "control", 2.0),
        "close_socket",
    ]


def test_non_finite_action_holds_measured_pose_without_forwarding_the_action():
    left = np.array([0.2, 0.1, 0.0, -0.1, -0.2, -0.3, 0.8])
    right = np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
    runtime, factory = _runtime(hardware=_hardware(left, right))
    runtime.connect_followers()
    invalid = np.concatenate([left, right])
    invalid[3] = np.nan

    result = runtime.command(invalid)

    assert result.rejection_reason == "non_finite_action"
    assert not result.clipped
    np.testing.assert_allclose(result.accepted, np.concatenate([left, right]))
    assert factory.followers[0].commands == []
    assert factory.followers[1].commands == []
    assert [backend.hold_calls for backend in factory.followers] == [2, 2]


def test_stale_feedback_holds_both_followers_without_writing_targets():
    runtime, factory = _runtime(config=DualYamJointEnvConfig(feedback_timeout_s=0.25))
    runtime.connect_followers()
    factory.followers[0].timestamp_s = 99.0

    with pytest.raises(RuntimeError, match="stale YAM follower feedback"):
        runtime.command(np.zeros(14))

    assert factory.followers[0].commands == []
    assert factory.followers[1].commands == []
    assert [backend.hold_calls for backend in factory.followers] == [2, 2]


def test_close_releases_each_backend_once_in_safe_reverse_order():
    runtime, factory = _runtime()
    runtime.connect_followers()
    runtime.connect_leaders()

    runtime.close()
    runtime.close()

    assert [backend.close_calls for backend in factory.followers] == [1, 1]
    assert [backend.close_calls for backend in factory.leaders] == [1, 1]
    close_order = [name for event, name in factory.events if event == "close"]
    assert close_order == [
        "right_leader",
        "left_leader",
        "right_follower",
        "left_follower",
    ]


def test_i2rt_backends_select_safe_role_specific_startup_modes(monkeypatch):
    startup_modes = []

    class _Robot:
        def get_robot_info(self):
            return {"kp": np.ones(6), "kd": np.ones(6)}

        def update_kp_kd(self, *, kp, kd):
            del kp, kd

    def build_robot(device_config, *, zero_gravity_mode):
        del device_config
        startup_modes.append(zero_gravity_mode)
        return _Robot()

    monkeypatch.setattr(i2rt_backend, "_build_yam", build_robot)
    device = SimpleNamespace(bilateral_kp=0.0)

    i2rt_backend.I2RTYamFollower(device).connect()
    i2rt_backend.I2RTYamLeader(device).connect()

    assert startup_modes == [False, True]


@pytest.mark.parametrize(
    ("gripper_invert", "expected_gripper"),
    [(False, 0.2), (True, 0.8)],
)
def test_i2rt_leader_maps_released_trigger_to_open_by_default(
    monkeypatch, gripper_invert, expected_gripper
):
    class _MotorChain:
        def get_same_bus_device_states(self):
            return [SimpleNamespace(position=0.8, io_inputs=(False, False))]

    class _Robot:
        motor_chain = _MotorChain()
        _joint_state = SimpleNamespace(timestamp=100.0)

        def get_robot_info(self):
            return {"kp": np.ones(6), "kd": np.ones(6)}

        def update_kp_kd(self, *, kp, kd):
            del kp, kd

        def get_joint_pos(self):
            return np.zeros(6)

    monkeypatch.setattr(
        i2rt_backend,
        "_build_yam",
        lambda device_config, *, zero_gravity_mode: _Robot(),
    )
    device = SimpleNamespace(
        bilateral_kp=0.0,
        gripper_invert=gripper_invert,
    )
    leader = i2rt_backend.I2RTYamLeader(device)
    leader.connect()

    state = leader.read_state()

    assert state.arm.gripper_position == pytest.approx(expected_gripper)


def test_i2rt_leader_position_command_temporarily_restores_native_gains(monkeypatch):
    events = []
    native_kp = np.arange(1, 7, dtype=np.float64)
    native_kd = np.arange(11, 17, dtype=np.float64)
    limits = np.column_stack([np.full(6, -1.0), np.full(6, 1.0)])

    class _Robot:
        def get_robot_info(self):
            return {
                "kp": native_kp.copy(),
                "kd": native_kd.copy(),
                "joint_limits": limits.copy(),
            }

        def update_kp_kd(self, *, kp, kd):
            events.append(("gains", np.asarray(kp).copy(), np.asarray(kd).copy()))

        def command_joint_pos(self, joints):
            events.append(("command", np.asarray(joints).copy()))

        def enter_gravity_comp_idle(self):
            events.append(("idle",))

    monkeypatch.setattr(
        i2rt_backend,
        "_build_yam",
        lambda device_config, *, zero_gravity_mode: _Robot(),
    )
    device = SimpleNamespace(
        bilateral_kp=0.25,
        gripper_invert=False,
    )
    leader = i2rt_backend.I2RTYamLeader(device)
    leader.connect()

    target = np.linspace(-0.3, 0.3, 6)
    leader.command(target)
    leader.command_feedback(np.full(6, 0.2))
    leader.release_feedback()

    np.testing.assert_allclose(events[0][1], native_kp * 0.25)
    np.testing.assert_allclose(events[0][2], np.zeros(6))
    np.testing.assert_allclose(events[1][1], native_kp)
    np.testing.assert_allclose(events[1][2], native_kd)
    np.testing.assert_allclose(events[2][1], target)
    np.testing.assert_allclose(events[3][1], native_kp * 0.25)
    np.testing.assert_allclose(events[3][2], np.zeros(6))
    np.testing.assert_allclose(events[4][1], np.full(6, 0.2))
    np.testing.assert_allclose(events[5][1], native_kp * 0.25)
    np.testing.assert_allclose(events[5][2], np.zeros(6))
    assert events[6] == ("idle",)
    np.testing.assert_allclose(leader.joint_limits(), limits)
