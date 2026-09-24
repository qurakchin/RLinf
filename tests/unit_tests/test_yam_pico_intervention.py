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

"""YAM VR controller, episode, and accepted-action contracts with mock hardware.

The station under test is the real one: the ``yam_pico`` device, the shared
intervention wrapper, and the YAM PICO episode wrapper, composed the way
``WrapperStack`` composes them. Only the hardware behind the device is
replaced -- the controllers report readings the test writes, and IK answers
with poses the test chooses.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rlinf.envs.real.wrappers.teleop.composed import ComposedTeleop
from rlinf.envs.real.wrappers.teleop.facts import EnvFacts
from rlinf.envs.real.wrappers.teleop.intervention import TeleopIntervention
from rlinf.envs.real.wrappers.teleop.layout import action_spec
from rlinf.envs.real.yam.config import YamPicoConfig
from rlinf.envs.real.yam.dual_yam_joint_env import DualYamJointEnv
from rlinf.envs.real.yam.kinematics import YamIKResult
from rlinf.envs.real.yam.pico_episode import YamPicoEpisode
from rlinf.robotics.parts.teleop.group import TeleopEntry, TeleopGroup
from rlinf.robotics.parts.teleop.yam_pico import SIDES, YamPico


class Arm:
    """A controller stub that answers with an action instead of mapping a pose.

    ``_YamPicoArm`` keeps the pose the grip was anchored to and turns the
    operator's motion into a scaled spatial error. That mapping has its own
    tests below; the station tests replace it so they can state the action
    they mean directly.
    """

    def __init__(self) -> None:
        self.ready = True
        self.held = False
        self.buttons: dict[str, bool] = {}
        self.action = np.array([0.5, 0, 0, 0, 0, 0, 0.0])
        self.reference = False
        self.resets = 0
        self.stopped = False

    def read(self) -> dict[str, object]:
        return {
            "held": self.held,
            "ready": self.ready,
            "calibrated": True,
            "control_value": float(self.held),
        }

    def read_buttons(self) -> dict[str, bool]:
        return self.buttons if self.ready else {}

    def reset_reference(self) -> None:
        self.reference = False
        self.resets += 1

    def command(
        self,
        reading,
        tcp_pose,
        action_scale,
        *,
        gripper_enabled=True,
        clip_motion=True,
    ):
        """Return the action this stub was told to produce for one reading."""
        del tcp_pose, action_scale, gripper_enabled, clip_motion
        held = bool(reading.get("held", False))
        action = self.action.copy()
        if held and not self.reference:
            # A freshly closed grip is anchored where the arm already stands.
            action[:6] = 0
        self.reference = held
        info = {
            "pico_active": held,
            "pico_ready": self.ready,
            "pico_calibrated": True,
            "pico_control_value": float(held),
        }
        return action.astype(np.float32), held and self.ready, info

    def stop(self) -> None:
        self.stopped = True


class Kinematics:
    """IK that answers with the target pose itself, unless told to fail."""

    def __init__(self) -> None:
        self.fail = False
        self.targets: list[np.ndarray] = []

    def fk(self, q, gripper):
        pose = np.eye(4)
        pose[:3, 3] = q[:3]
        pose[:3, :3] = Rotation.from_rotvec(q[3:6]).as_matrix()
        return pose

    def solve(self, target, seed, gripper):
        self.targets.append(target.copy())
        q = np.concatenate(
            [target[:3, 3], Rotation.from_matrix(target[:3, :3]).as_rotvec()]
        )
        return YamIKResult(
            not self.fail, q, 0.0, 0.0, 0.0, "injected_failure" if self.fail else None
        )


def build_station(*, arms=None, kinematics=None, **base_overrides):
    """Compose the PICO station the shared wrapper stack builds.

    Returns a namespace holding the wrapper a caller steps, the device under
    it, the base environment, and the injected controllers and IK.
    """
    base = DualYamJointEnv(
        {
            "is_dummy": True,
            "image_height": 8,
            "image_width": 8,
            "manual_episode_control_only": True,
            **base_overrides,
        }
    )
    base._pace = lambda: None
    config = YamPicoConfig(wait_for_record_button=False, button_debounce_s=0.0)
    spec = action_spec(base)
    facts = EnvFacts.about(base, spec.layout, spec.kinds)
    device = YamPico(
        config,
        joint_step_limits=facts.joint_step_limits,
        joint_lower=facts.joint_limit_min,
        joint_upper=facts.joint_limit_max,
        experts=arms if arms is not None else {side: Arm() for side in SIDES},
        kinematics=(
            kinematics
            if kinematics is not None
            else {side: Kinematics() for side in SIDES}
        ),
    )
    device.connect()
    group = TeleopGroup([TeleopEntry(device)], available=facts.kinds)
    composed = ComposedTeleop(group, facts.layout, timeout=group.hold_window)
    return SimpleNamespace(
        env=YamPicoEpisode(
            TeleopIntervention(base, composed, mark_flag=base.TELEOP_MARK_FLAG),
            config,
        ),
        device=device,
        base=base,
        arms=device._experts,
        kinematics=device._kinematics,
        layout=dict(facts.layout),
    )


@pytest.fixture
def station():
    composed = build_station()
    composed.env.reset()
    composed.env.step(np.zeros(14))  # Observe released grips, prime button edges.
    yield composed
    composed.env.close()


def press(station, button):
    """Release one frame, then hold, producing a button edge on the second."""
    station.arms["right"].buttons = {}
    station.env.step(np.zeros(14))
    station.arms["right"].buttons = {button: True}
    return station.env.step(np.zeros(14))


def map_state(station, measured):
    """Map one device reading against a measured state into a joint target."""
    teleop = station.device.drive(
        {"joint_positions": np.asarray(measured, dtype=np.float64)}
    )
    target = np.asarray(measured, dtype=np.float64).copy()
    for name, value in teleop.parts.items():
        target[station.layout[name]] = np.asarray(value, dtype=np.float64)
    return target, teleop.info


def test_single_arm_first_engagement_and_gripper_latch(station):
    arm = station.arms["left"]
    arm.held = True
    _, _, _, _, first = station.env.step(np.ones(14))
    np.testing.assert_allclose(first["intervene_action"][:6], 0.0)
    arm.action[6] = 1.0
    _, _, _, _, info = station.env.step(np.ones(14))
    assert info["intervene_action"][0] > 0
    assert info["intervene_action"][6] == 1.0
    np.testing.assert_array_equal(info["intervene_action"][7:], np.zeros(7))
    arm.action[6] = 0.0
    assert station.env.step(np.zeros(14))[-1]["intervene_action"][6] == 1.0
    arm.held = False
    held = station.base.get_hold_action()
    np.testing.assert_allclose(
        station.env.step(np.zeros(14))[-1]["intervene_action"], held
    )


@pytest.mark.parametrize("side,index", [("left", 0), ("right", 7)])
def test_inactive_arm_holds_fixed_joints_and_recaptures_release_pose(
    station, monkeypatch, side, index
):
    measured = station.base.get_hold_action().astype(np.float64)
    initial = measured.copy()
    joints = slice(index, index + 6)
    monkeypatch.setattr(
        station.base, "get_joint_positions", lambda: measured.copy().astype(np.float32)
    )

    # Startup idle keeps its reference even if the encoder reports drift.
    measured[joints] += 0.02
    info = station.env.step(np.zeros(14))[-1]
    np.testing.assert_allclose(info["intervene_action"][joints], initial[joints])
    assert not info["pico_active"]

    # Taking control uses the current pose, not the old idle target.
    station.arms[side].held = True
    info = station.env.step(np.zeros(14))[-1]
    np.testing.assert_allclose(info["intervene_action"][joints], measured[joints])
    assert info[side]

    # Releasing the grip captures a new reference once for this arm only.
    measured[joints] += 0.02
    released = measured.copy()
    station.arms[side].held = False
    station.env.step(np.zeros(14))
    for _ in range(3):
        measured[joints] += 0.01
        info = station.env.step(np.zeros(14))[-1]
        np.testing.assert_allclose(info["intervene_action"][joints], released[joints])
        other_joints = slice(7, 13) if index == 0 else slice(0, 6)
        np.testing.assert_allclose(
            info["intervene_action"][other_joints], initial[other_joints]
        )


def test_data_fault_recovery_replaces_old_idle_reference(station, monkeypatch):
    measured = station.base.get_hold_action().astype(np.float64)
    monkeypatch.setattr(
        station.base, "get_joint_positions", lambda: measured.copy().astype(np.float32)
    )
    measured[:6] += 0.02
    station.arms["right"].ready = False
    info = station.env.step(np.zeros(14))[-1]
    assert info["yam_pico_fault"]
    np.testing.assert_allclose(info["intervene_action"], measured)

    # Recovery captures the actual pose instead of restoring a pre-fault target.
    measured[:6] += 0.02
    recovered = measured.copy()
    station.arms["right"].ready = True
    assert station.env.step(np.zeros(14))[-1]["yam_pico_fault"] is None
    measured[:6] += 0.02
    info = station.env.step(np.zeros(14))[-1]
    np.testing.assert_allclose(info["intervene_action"][:6], recovered[:6])


@pytest.mark.parametrize("fault", ["stale", "nan"])
def test_fault_holds_both_and_requires_release_before_reengagement(station, fault):
    for arm in station.arms.values():
        arm.held = True
    station.env.step(np.zeros(14))
    held = station.base.get_hold_action()
    if fault == "stale":
        station.arms["right"].ready = False
    else:
        station.arms["right"].action[0] = np.nan
    info = station.env.step(np.zeros(14))[-1]
    assert info["yam_pico_fault"]
    np.testing.assert_allclose(info["intervene_action"], held)
    station.arms["right"].ready = True
    station.arms["right"].action[0] = 0.5
    assert station.env.step(np.zeros(14))[-1]["yam_pico_fault"]
    for arm in station.arms.values():
        arm.held = False
    assert station.env.step(np.zeros(14))[-1]["yam_pico_fault"] is None
    station.arms["left"].held = True
    np.testing.assert_allclose(
        station.env.step(np.zeros(14))[-1]["intervene_action"], held
    )


@pytest.mark.parametrize("side", ["left", "right"])
def test_ik_failure_holds_only_failed_arm_and_retries_without_release(station, side):
    for arm in station.arms.values():
        arm.held = True
    station.env.step(np.zeros(14))
    press(station, "right_menu_button")
    station.arms["right"].buttons = {}
    held = station.base.get_hold_action().copy()
    resets = {name: arm.resets for name, arm in station.arms.items()}
    for arm in station.arms.values():
        arm.action[6] = 1.0
    kin = station.kinematics[side]
    failed_index = 0 if side == "left" else 7
    healthy_index = 7 - failed_index
    healthy_side = "right" if side == "left" else "left"
    kin.fail = True
    solves = len(kin.targets)
    for attempt in range(3):
        _, reward, done, _, info = station.env.step(np.zeros(14))
        assert len(kin.targets) == solves + attempt + 1
        assert info["yam_pico_fault"] == f"{side}:ik:injected_failure"
        assert info["pico_active"] and info[healthy_side] and not info[side]
        assert info["pre_record"] and reward == 0 and not done
        assert info["record_reset"] == (attempt == 0)
        np.testing.assert_allclose(
            info["intervene_action"][failed_index : failed_index + 7],
            held[failed_index : failed_index + 7],
        )
        assert info["intervene_action"][healthy_index] > held[healthy_index]
        assert info["intervene_action"][healthy_index + 6] == 1.0
        for name, arm in station.arms.items():
            assert arm.held and arm.reference
            assert arm.resets == resets[name]

    kin.fail = False
    for arm in station.arms.values():
        arm.action[6] = 0.0
    info = station.env.step(np.zeros(14))[-1]
    assert info["yam_pico_fault"] is None and info["pico_active"]
    assert info["intervene_action"][0] > held[0]
    assert info["intervene_action"][7] > held[7]
    # An open command from a rejected frame must not leak into the retry.
    assert info["intervene_action"][failed_index + 6] == held[failed_index + 6]
    assert info["intervene_action"][healthy_index + 6] == 1.0


def test_both_ik_failures_hold_both_and_report_each_reason(station):
    for arm in station.arms.values():
        arm.held = True
    station.env.step(np.zeros(14))
    held = station.base.get_hold_action().copy()
    for kin in station.kinematics.values():
        kin.fail = True
    info = station.env.step(np.zeros(14))[-1]
    np.testing.assert_allclose(info["intervene_action"], held)
    assert not info["pico_active"]
    assert info["left_ik_fault"] == info["right_ik_fault"] == "injected_failure"


def test_data_loss_during_ik_retry_still_requires_grip_release(station):
    station.arms["left"].held = True
    station.env.step(np.zeros(14))
    station.kinematics["left"].fail = True
    assert station.env.step(np.zeros(14))[-1]["yam_pico_fault"].startswith("left:ik:")
    station.arms["right"].ready = False
    assert station.env.step(np.zeros(14))[-1]["yam_pico_fault"] == "right:unavailable"
    station.kinematics["left"].fail = False
    station.arms["right"].ready = True
    assert station.env.step(np.zeros(14))[-1]["yam_pico_fault"] == "right:unavailable"
    station.arms["left"].held = False
    assert station.env.step(np.zeros(14))[-1]["yam_pico_fault"] is None


def test_record_edges_abort_and_success_preserve_only_manual_boundary(station):
    station.arms["left"].held = True
    info = press(station, "right_menu_button")[-1]
    assert info["record_reset"] and not info["pre_record"]
    assert not station.env.step(np.zeros(14))[2]  # Held button cannot end again.
    result = press(station, "right_menu_button")
    assert result[1] == 1 and result[2] and result[-1]["success"]
    count = station.arms["left"].resets
    station.env.reset()
    assert station.arms["left"].resets == count
    station.env.reset()  # Early reset must clear the reference.
    assert station.arms["left"].resets > count
    station.arms["left"].held = False
    station.env.step(np.zeros(14))
    press(station, "right_menu_button")
    info = press(station, "left_menu_button")[-1]
    assert info["pre_record"] and info["record_reset"] and not info["success"]


def test_fault_discards_active_recording(station):
    press(station, "right_menu_button")
    station.arms["left"].ready = False
    _, reward, done, _, info = station.env.step(np.zeros(14))
    assert info["record_reset"] and info["pre_record"]
    assert not done and reward == 0.0


def test_recorded_action_uses_runtime_acceptance(station, monkeypatch):
    original = station.env.env.step

    def clipped(action):
        obs, reward, done, truncated, info = original(action)
        info["accepted_action"] = np.full(14, 0.123, dtype=np.float32)
        return obs, reward, done, truncated, info

    monkeypatch.setattr(station.env.env, "step", clipped)
    np.testing.assert_allclose(
        station.env.step(np.zeros(14))[-1]["intervene_action"], 0.123
    )


@pytest.mark.parametrize("joint_step", [0.05, 0.08])
def test_large_valid_ik_target_is_interpolated_and_recorded(monkeypatch, joint_step):
    station = build_station(max_joint_delta=joint_step)
    try:
        station.env.reset()
        station.env.step(np.zeros(14))
        station.arms["left"].held = True
        station.env.step(np.zeros(14))
        before = station.base.get_hold_action().copy()

        def large_target(target, seed, gripper):
            del target, gripper
            return YamIKResult(
                True, seed + np.array([0.2, 0.1, 0.05, 0.1, 0, -0.1]), 0.0, 0.0, 0.0
            )

        monkeypatch.setattr(station.kinematics["left"], "solve", large_target)
        info = station.env.step(np.zeros(14))[-1]
        assert info["yam_pico_fault"] is None and info["pico_active"]
        np.testing.assert_allclose(
            info["intervene_action"][:6],
            before[:6] + joint_step * np.array([1, 0.5, 0.25, 0.5, 0, -0.5]),
        )
        np.testing.assert_allclose(info["intervene_action"][6:], before[6:])
        np.testing.assert_array_equal(info["intervene_action"], info["accepted_action"])
    finally:
        station.env.close()


def test_reset_wait_keeps_teleop_ticking_without_episode_steps(station):
    station.env.config.wait_for_record_button = True

    class Keyboard:
        count = 0

        def pop_pressed_keys(self):
            self.count += 1
            return ["r"] if self.count == 3 else []

        def close(self):
            pass

    keyboard = Keyboard()
    station.env._keyboard = keyboard
    _, info = station.env.reset()
    assert keyboard.count == 3
    assert not info["pre_record"]
    assert station.base._num_steps == 0


def test_device_build_refuses_combinations_its_runtime_cannot_honor():
    facts = EnvFacts(layout={}, kinds={}, joint_step_limits=(0.08,) * 6)
    with pytest.raises(ValueError, match="mutually exclusive"):
        YamPico.from_config(
            {"override_cfg": {"leader_intervention": {"enabled": True}}}, {}, facts
        )
    with pytest.raises(ValueError, match="enforce_runtime"):
        YamPico.from_config(
            {"override_cfg": {"enforce_runtime_joint_limits": False}}, {}, facts
        )
    with pytest.raises(ValueError, match="drives"):
        YamPico.from_config({}, {"drives": "left"}, facts)


def test_real_pico_arm_lifecycle_preserves_calibration(monkeypatch):
    from rlinf.robotics.parts.teleop.yam_pico import _YamPicoArm
    from rlinf.robotics.parts.transports.pico import PicoExpert

    monkeypatch.setattr(PicoExpert, "start", lambda self: None)
    arm = _YamPicoArm(hand="right", calibration={"enabled": False})
    arm._expert._latest_data = {
        "right_controller": {
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "grip": 1.0,
        },
        "buttons": {"A": True},
    }
    arm._expert._last_update_time = time.time()
    tcp = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])
    scale = np.array([0.01, 0.1, 1.0])
    action, engaged, _ = arm.command(arm.read(), tcp, scale)
    assert engaged
    np.testing.assert_allclose(action[:6], 0)
    assert arm.read_buttons()["A"]
    arm._expert._calibrated = True
    arm.reset_reference()
    assert arm._expert._calibrated and arm._ref_tcp_pos is None
    action, _, _ = arm.command(arm.read(), tcp, scale)
    np.testing.assert_allclose(action[:6], 0)
    arm.stop()


def test_configuration_rejects_ambiguous_and_invalid_settings():
    for config in (
        {"hand": "left"},
        {"control_threshold": float("nan")},
        {"record_button": "A"},
        {"left": {"hand": "right"}},
        {"ik": {"max_iters": 0}},
        {"rotation_delta_frame": "unsupported"},
        {"ik_backtrack_attempts": -1},
        {"ik_backtrack_attempts": True},
        {"ik_backtrack_attempts": 1.5},
    ):
        with pytest.raises(ValueError):
            YamPicoConfig(**config)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("success_attempt", [1, 2, 3, 4, None])
def test_ik_backtracking_scales_spatial_delta_and_preserves_seed(
    station, monkeypatch, side, success_attempt
):
    station.env.config.ik_backtrack_attempts = 3
    station.env.config.ik.max_solve_s = 1.0
    arm = np.array([0.1, 0.2, 0.3, 0.4, -0.3, 0.2, 0.5])
    original_arm = arm.copy()
    current = station.kinematics[side].fk(arm[:6], arm[6])
    target = current.copy()
    translation = np.array([0.2, -0.1, 0.3])
    rotation = np.array([0.3, -0.6, 0.9])
    target[:3, 3] += translation
    target[:3, :3] = Rotation.from_rotvec(rotation).as_matrix() @ current[:3, :3]
    original_target = target.copy()
    candidates = []

    def solve(candidate, seed, gripper):
        np.testing.assert_array_equal(seed, original_arm[:6])
        assert gripper == original_arm[6]
        candidates.append(candidate.copy())
        # Failed solver iterates must not become the next seed or mutate feedback.
        seed[:] = 99.0
        ok = len(candidates) == success_attempt
        return YamIKResult(
            ok, original_arm[:6].copy(), 0, 0, 0, None if ok else "not_converged"
        )

    monkeypatch.setattr(station.kinematics[side], "solve", solve)
    result, attempts, fraction, first_reason = station.device._solve_with_backtracking(
        side, target, current, arm
    )
    assert attempts == (success_attempt or 4)
    assert fraction == 0.5 ** (attempts - 1)
    assert result.success == (success_attempt is not None)
    assert first_reason == (None if success_attempt == 1 else "not_converged")
    for index, candidate in enumerate(candidates):
        factor = 0.5**index
        np.testing.assert_allclose(
            candidate[:3, 3], current[:3, 3] + factor * translation
        )
        delta = Rotation.from_matrix(candidate[:3, :3] @ current[:3, :3].T)
        np.testing.assert_allclose(delta.as_rotvec(), factor * rotation, atol=1e-12)
    np.testing.assert_array_equal(arm, original_arm)
    np.testing.assert_array_equal(target, original_target)


@pytest.mark.parametrize(
    "reason,retries",
    [
        ("not_converged", 3),
        ("residual", 3),
        ("joint_limits", 3),
        ("solve_timeout", 0),
        ("invalid_solution", 0),
        ("finger_motion", 0),
        ("seed_out_of_limits", 0),
        ("solver_error:ValueError", 0),
    ],
)
def test_ik_backtracking_only_retries_geometric_failures(
    station, monkeypatch, reason, retries
):
    station.env.config.ik_backtrack_attempts = 3
    station.env.config.ik.max_solve_s = 1.0
    monkeypatch.setattr(
        station.kinematics["left"],
        "solve",
        lambda *args: YamIKResult(False, np.zeros(6), 0, 0, 0, reason),
    )
    result, attempts, _, _ = station.device._solve_with_backtracking(
        "left", np.eye(4), np.eye(4), np.zeros(7)
    )
    assert not result.success
    assert attempts == 1 + retries
    assert result.reason == reason


@pytest.mark.parametrize("durations", [[0.04], [0.01, 0.021]])
def test_ik_backtracking_shares_budget_and_rejects_late_success(
    station, monkeypatch, durations
):
    import rlinf.robotics.parts.teleop.yam_pico as module

    station.env.config.ik_backtrack_attempts = 3
    now, calls = [0.0], []
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])

    def solve(*args):
        now[0] += durations[len(calls)]
        calls.append(True)
        ok = len(calls) == len(durations)
        return YamIKResult(
            ok,
            np.ones(6),
            0,
            0,
            durations[len(calls) - 1],
            None if ok else "not_converged",
        )

    monkeypatch.setattr(station.kinematics["left"], "solve", solve)
    result, attempts, _, _ = station.device._solve_with_backtracking(
        "left", np.eye(4), np.eye(4), np.zeros(7)
    )
    assert attempts == len(durations)
    assert not result.success
    assert result.reason == "solve_timeout"
    assert result.elapsed_s == pytest.approx(sum(durations))


@pytest.mark.parametrize("side,index", [("left", 0), ("right", 7)])
@pytest.mark.parametrize("recover", [True, False])
def test_ik_backtracking_dispatches_only_valid_reduced_targets(
    station, monkeypatch, side, index, recover
):
    station.env.config.ik_backtrack_attempts = 3
    station.env.config.ik.max_solve_s = 1.0
    for arm in station.arms.values():
        arm.held = True
    station.env.step(np.zeros(14))
    before = station.base.get_hold_action().copy()
    station.arms[side].action[0] = 4.0  # Full target is 20 mm ahead.
    original_solve = station.kinematics[side].solve
    calls = []

    def solve(target, seed, gripper):
        calls.append(target.copy())
        if recover and len(calls) == 3:
            return original_solve(target, seed, gripper)
        return YamIKResult(False, np.full(6, 99.0), 1, 1, 0, "not_converged")

    monkeypatch.setattr(station.kinematics[side], "solve", solve)
    info = station.env.step(np.zeros(14))[-1]
    assert info[f"{side}_ik_attempts"] == (3 if recover else 4)
    assert info[f"{side}_ik_initial_fault"] == "not_converged"
    assert info[f"{side}_ik_fault"] == (None if recover else "not_converged")
    expected = before[index : index + 6].copy()
    if recover:
        expected[0] += 0.005
        assert info[f"{side}_ik_target_fraction"] == 0.25
    np.testing.assert_allclose(
        info["accepted_action"][index : index + 6], expected, atol=1e-7
    )
    other = 7 if index == 0 else 0
    assert info["accepted_action"][other] > before[other]
    assert station.device.fault is None  # Exhaustion does not latch both arms.
    assert station.arms[side].reference


def test_two_real_subscribers_receive_and_close_independently():
    zmq = pytest.importorskip("zmq")
    from rlinf.robotics.parts.transports.pico import PicoExpert

    # ``ipc://`` addresses are capped at ``sizeof(sun_path)`` (103 bytes), and
    # pytest's own ``tmp_path`` on macOS already exceeds that.
    directory = tempfile.mkdtemp(prefix="yam-pico-")
    address = f"ipc://{directory}/pico.ipc"
    context = zmq.Context()
    publisher = None
    experts = []
    try:
        publisher = context.socket(zmq.PUB)
        publisher.bind(address)
        for side in ("left", "right"):
            experts.append(
                PicoExpert(
                    zmq_addr=address,
                    hand=side,
                    timeout_ms=20,
                    calibration={"enabled": False},
                )
            )
        assert experts[0]._socket is not experts[1]._socket
        assert experts[0]._thread is not experts[1]._thread
        message = {
            "headset_pose": [0, 1, 0, 0, 0, 0, 1],
            "buttons": {"X": True},
            "left_controller": {
                "grip": 1,
                "position": [0, 0, 0],
                "orientation": [0, 0, 0, 1],
            },
            "right_controller": {
                "grip": 0,
                "position": [0, 0, 0],
                "orientation": [0, 0, 0, 1],
            },
        }
        deadline = time.monotonic() + 2
        while not all(e.ready for e in experts) and time.monotonic() < deadline:
            publisher.send_json(message)
            time.sleep(0.01)
        assert all(e.ready for e in experts)
        # Only the gripping hand is driven; each reader sees its own grip.
        assert experts[0].get_reading()["held"]
        assert not experts[1].get_reading()["held"]
        assert experts[0].get_buttons()["X"]
        threads = [e._thread for e in experts]
    finally:
        for expert in experts:
            expert.stop()
        if publisher is not None:
            publisher.close(linger=0)
        context.term()
        shutil.rmtree(directory, ignore_errors=True)
    assert all(not thread.is_alive() for thread in threads)


def test_rotation_delta_uses_base_frame_left_multiplication(station):
    arm = station.arms["left"]
    arm.held = True
    station.env.step(np.zeros(14))
    arm.action[:] = [0, 0, 0, 1, 0, 0, 0]
    before = station.base.get_hold_action()[:6]
    station.env.step(np.zeros(14))
    expected = (
        Rotation.from_rotvec([station.env.config.max_rotation_delta, 0, 0]).as_matrix()
        @ Rotation.from_rotvec(before[3:6]).as_matrix()
    )
    np.testing.assert_allclose(station.kinematics["left"].targets[-1][:3, :3], expected)


def test_runtime_rejection_aborts_recording_and_reports_accepted_hold(
    station, monkeypatch
):
    press(station, "right_menu_button")
    original = station.env.env.step

    def reject(action):
        result = original(station.base.get_hold_action())
        result[-1]["action_rejected"] = "measured_joint_out_of_limits"
        return result

    monkeypatch.setattr(station.env.env, "step", reject)
    info = station.env.step(np.zeros(14))[-1]
    assert info["record_reset"] and info["pre_record"]
    assert info["yam_pico_fault"].startswith("runtime:")
    np.testing.assert_array_equal(info["accepted_action"], info["intervene_action"])


def test_device_rolls_back_when_ik_setup_fails(monkeypatch, tmp_path):
    arms = {side: Arm() for side in SIDES}

    def fail(**kwargs):
        del kwargs
        raise ValueError("invalid model")

    monkeypatch.setattr("rlinf.envs.real.yam.kinematics.YamKinematicsAdapter", fail)
    station = build_station(arms=arms)
    station.device.disconnect()
    device = YamPico(
        YamPicoConfig(),
        joint_step_limits=(0.08,) * 6,
        experts=arms,
    )
    with pytest.raises(ValueError, match="invalid model"):
        device.connect()
    assert not device.is_connected
    assert all(arm.stopped for arm in arms.values())


def test_unexpected_control_exception_closes_followers_and_arms(station, monkeypatch):
    def fail(*args):
        raise RuntimeError("injected FK error")

    monkeypatch.setattr(station.kinematics["left"], "fk", fail)
    with pytest.raises(RuntimeError, match="injected FK error"):
        station.env.step(np.zeros(14))
    assert station.base._closed
    assert all(arm.stopped for arm in station.arms.values())


class PackedObservations(gym.ObservationWrapper):
    def observation(self, obs):
        return {
            "states": obs["state"]["joint_position"],
            "main_images": obs["frames"]["top_rgb"],
            "extra_view_images": np.stack(
                [obs["frames"]["left_rgb"], obs["frames"]["right_rgb"]]
            ),
            "task_descriptions": "pick_block",
        }


@pytest.mark.parametrize("write_to_disk", [False, True])
def test_two_lerobot_episodes_exclude_preview_and_discarded_frames(
    station, tmp_path, monkeypatch, write_to_disk
):
    from rlinf.envs.wrappers.collect_episode import CollectEpisode

    collector = CollectEpisode(
        PackedObservations(station.env),
        str(tmp_path),
        export_format="lerobot",
        only_success=True,
    )
    episodes = []
    if write_to_disk:
        pytest.importorskip("lerobot")
        from rlinf.data.storage.lerobot.writer import LeRobotDatasetWriter

        original_create = LeRobotDatasetWriter.create

        def create(self, **kwargs):
            kwargs.update(image_writer_processes=0, image_writer_threads=1)
            return original_create(self, **kwargs)

        monkeypatch.setattr(LeRobotDatasetWriter, "create", create)
        original_write = collector._write_lerobot_episode

        def write(episode):
            original_write(episode)
            episodes.append(episode)

        monkeypatch.setattr(collector, "_write_lerobot_episode", write)
    else:
        monkeypatch.setattr(collector, "_write_lerobot_episode", episodes.append)
    try:
        collector.reset()
        station.arms["left"].held = False
        collector.step(np.zeros(14))

        def edge(button):
            station.arms["right"].buttons = {}
            collector.step(np.zeros(14))
            station.arms["right"].buttons = {button: True}
            return collector.step(np.zeros(14))

        edge("right_menu_button")
        collector.step(np.zeros(14))
        edge("left_menu_button")
        assert collector._buffers[0]["actions"] == []
        for _ in range(2):
            edge("right_menu_button")
            before = station.base.get_hold_action().copy()
            station.arms["left"].held = True
            _, _, _, _, info = collector.step(np.zeros(14))
            expected = info["accepted_action"].copy()
            edge("right_menu_button")
            collector._wait_futures()
            frame = episodes[-1][0]
            np.testing.assert_allclose(frame["state"], before)
            np.testing.assert_allclose(frame["actions"], expected)
            assert frame["actions"].shape == (14,)
            assert frame["image"].shape == (8, 8, 3)
            assert "extra_view_image-1" in frame
            assert episodes[-1][-1]["done"].all()
            collector.reset()
        assert len(episodes) == 2
    finally:
        collector.close()
    if write_to_disk:
        import json

        import pyarrow.parquet as pq

        shard = tmp_path / "rank_0/id_0"
        metadata = json.loads((shard / "meta/info.json").read_text())
        assert metadata["total_episodes"] == 2
        tables = [
            pq.read_table(path) for path in sorted(shard.glob("data/**/*.parquet"))
        ]
        assert len(tables) == 2
        for table, episode in zip(tables, episodes, strict=True):
            np.testing.assert_allclose(
                table["actions"].to_pylist(), [frame["actions"] for frame in episode]
            )


@pytest.mark.parametrize("hand", ["left", "right"])
@pytest.mark.parametrize("initial_yaw_deg", [0.0, 90.0])
@pytest.mark.parametrize("motion", [0.001, 0.2])
def test_yam_vr_target_stays_anchored_at_the_grip_pose(
    station, monkeypatch, hand, initial_yaw_deg, motion
):
    """The operator's motion adds to where the arm stood at the grip edge.

    A target composed from the *current* feedback instead would stall while the
    position controller catches up, which is what ``motion=0.2`` reproduces.
    """
    from rlinf.robotics.parts.teleop.yam_pico import _YamPicoArm
    from rlinf.robotics.parts.transports.pico import PicoExpert

    monkeypatch.setattr(PicoExpert, "start", lambda self: None)
    arm = _YamPicoArm(**station.env.config.expert_kwargs(hand))
    station.arms[hand] = arm
    index = 0 if hand == "left" else 7
    reference = Rotation.from_euler("z", initial_yaw_deg, degrees=True)
    controller = {
        "position": [0.0, 1.0, 0.0],
        "orientation": reference.as_quat().tolist(),
        "grip": 1.0,
    }
    message = {
        "headset_pose": [0, 1.6, 0, 0, 0, 0, 1],
        f"{hand}_controller": controller,
    }
    anchor = None

    try:
        for step in range(3):
            if step:
                controller["position"] = [
                    motion * step,
                    1 + motion * step / 10,
                    -motion * step,
                ]
                controller["orientation"] = (
                    (
                        Rotation.from_rotvec([motion * step, -motion / 3, motion / 2])
                        * reference
                    )
                    .as_quat()
                    .tolist()
                )
            arm._expert._latest_data = message
            arm._expert._last_update_time = time.time()
            reading = arm._expert.get_reading()
            assert reading["held"]
            if anchor is None:
                q = station.base.get_hold_action()[index : index + 7]
                anchor = station.kinematics[hand].fk(q[:6], q[6])[:3, 3].copy()
            _, _, _, _, info = station.env.step(np.zeros(14))
            assert info["yam_pico_fault"] is None
            target = station.kinematics[hand].targets[-1]
            np.testing.assert_allclose(
                target[:3, 3], anchor + reading["position_delta"], atol=1e-7
            )
    finally:
        arm.stop()


def test_composed_tcp_target_preserves_the_original_equations():
    from rlinf.robotics.parts.teleop.yam_pico import _delta_to_tcp_pose

    current_rot = Rotation.from_euler("xyz", [0.3, -0.4, 0.5])
    tcp = np.r_[0.4, -0.2, 0.3, current_rot.as_quat()]
    action = np.array([1, 1, -0.5, 1, -1, 1, 0], dtype=np.float32)
    scale = np.array([0.005, 0.03, 1.0])
    target = _delta_to_tcp_pose(action, tcp, scale)
    # No extra vector-norm clipping of the translation.
    np.testing.assert_allclose(target[:3], tcp[:3] + [0.005, 0.005, -0.0025], atol=1e-9)
    expected_rot = (
        Rotation.from_rotvec(np.array([1, -1, 1]) / np.sqrt(3) * 0.03) * current_rot
    )
    np.testing.assert_allclose(
        Rotation.from_quat(target[3:]).as_matrix(), expected_rot.as_matrix()
    )


@pytest.mark.parametrize("initial_yaw_deg", [0.0, 90.0])
def test_original_controller_local_delta_and_axis_clipping(
    monkeypatch, initial_yaw_deg
):
    from rlinf.robotics.parts.teleop.yam_pico import _YamPicoArm
    from rlinf.robotics.parts.transports.pico import R_PICO_TO_WORLD, PicoExpert

    monkeypatch.setattr(PicoExpert, "start", lambda self: None)
    kwargs = YamPicoConfig().expert_kwargs("right")
    kwargs.update(
        position_scale=1.0,
        rotation_scale=1.0,
        rotation_delta_frame="controller_local",
        calibration={"enabled": False},
    )
    arm = _YamPicoArm(**kwargs)
    ref = Rotation.from_euler("z", initial_yaw_deg, degrees=True)
    controller = {
        "position": [0, 0, 0],
        "orientation": ref.as_quat().tolist(),
        "grip": 1.0,
    }
    arm._expert._latest_data = {"right_controller": controller}
    arm._expert._last_update_time = time.time()
    tcp = np.array([0.4, 0, 0.3, 0, 0, 0, 1.0])
    scale = np.array([0.005, 0.03, 1.0])
    arm.command(arm.read(), tcp, scale)
    controller["position"] = (R_PICO_TO_WORLD.T @ np.array([0.05, 0.005, 0])).tolist()
    current = Rotation.from_rotvec([0.01, 0, 0]) * ref
    controller["orientation"] = current.as_quat().tolist()
    action, _, _ = arm.command(arm.read(), tcp, scale)
    np.testing.assert_allclose(action[:3], [1, 1, 0])
    local_delta = (ref.inv() * current).as_rotvec()
    expected = np.array([-local_delta[2], -local_delta[0], local_delta[1]])
    np.testing.assert_allclose(action[3:6] * scale[1], expected, atol=1e-8)
    arm.stop()


@pytest.mark.parametrize("hand", ["left", "right"])
def test_full_vr_lift_target_survives_stalled_feedback(station, monkeypatch, hand):
    from rlinf.robotics.parts.teleop.yam_pico import _YamPicoArm
    from rlinf.robotics.parts.transports.pico import PicoExpert

    monkeypatch.setattr(PicoExpert, "start", lambda self: None)
    station.env.config.position_scale = 1.0
    arm = _YamPicoArm(**station.env.config.expert_kwargs(hand))
    station.arms[hand] = arm
    measured = station.base.get_hold_action().astype(np.float64)
    monkeypatch.setattr(
        station.base, "get_joint_positions", lambda: measured.copy().astype(np.float32)
    )
    controller = {
        "position": [0.0, 1.0, 0.0],
        "orientation": [0, 0, 0, 1],
        "grip": 1.0,
    }
    arm._expert._latest_data = {
        "headset_pose": [0, 1.6, 0, 0, 0, 0, 1],
        f"{hand}_controller": controller,
    }
    arm._expert._last_update_time = time.time()
    map_state(station, measured)  # Grip captures the measured TCP reference.
    start = station.kinematics[hand].targets[-1].copy()
    controller["position"][1] += 0.15  # Raw VR +Y maps to arm +Z.
    for _ in range(3):
        arm._expert._last_update_time = time.time()
        _, info = map_state(station, measured)
        assert info["yam_pico_fault"] is None
        target = station.kinematics[hand].targets[-1]
        np.testing.assert_allclose(
            target[:3, 3] - start[:3, 3], [0, 0, 0.15], atol=1e-7
        )


@pytest.mark.parametrize("hand", ["left", "right"])
@pytest.mark.parametrize("initial_hand_deg", [[0, 0, 0], [0, 0, 90], [35, -25, 70]])
@pytest.mark.parametrize("head_yaw_deg,robot_yaw", [(0, 0), (90, 0.4)])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_yam_spatial_rotation_matches_translation_axes(
    station, monkeypatch, hand, initial_hand_deg, head_yaw_deg, robot_yaw, axis
):
    from rlinf.robotics.parts.teleop.yam_pico import _YamPicoArm
    from rlinf.robotics.parts.transports.pico import R_PICO_TO_WORLD, PicoExpert

    monkeypatch.setattr(PicoExpert, "start", lambda self: None)
    station.env.config.rotation_scale = 1.0
    getattr(station.env.config, hand)["operator_to_robot_yaw"] = robot_yaw
    arm = _YamPicoArm(**station.env.config.expert_kwargs(hand))
    station.arms[hand] = arm
    index = 0 if hand == "left" else 7
    measured = station.base.get_hold_action().astype(np.float64)
    measured[index : index + 6] = [0.2, 0.3, 0.4, 0.3, -0.4, 0.5]
    monkeypatch.setattr(
        station.base, "get_joint_positions", lambda: measured.copy().astype(np.float32)
    )
    reference = station.kinematics[hand].fk(
        measured[index : index + 6], measured[index + 6]
    )
    hand_ref = Rotation.from_euler("xyz", initial_hand_deg, degrees=True)
    head = Rotation.from_euler("y", head_yaw_deg, degrees=True)
    controller = {
        "position": [0, 1, 0],
        "orientation": hand_ref.as_quat().tolist(),
        "grip": 1.0,
    }
    packet = {
        "headset_pose": [0, 1.6, 0, *head.as_quat()],
        f"{hand}_controller": controller,
    }
    arm._expert._latest_data = packet
    arm._expert._last_update_time = time.time()
    map_state(station, measured)
    # Independent change-of-basis oracle. Head raw +Y yaw becomes operator +Z.
    basis = (
        Rotation.from_euler("z", robot_yaw - np.deg2rad(head_yaw_deg)).as_matrix()
        @ R_PICO_TO_WORLD
    )
    for degrees in (10.0, -10.0):
        vector = np.eye(3)[axis] * np.deg2rad(degrees)
        delta = Rotation.from_rotvec(vector).as_matrix()
        raw_delta = basis.T @ delta @ basis
        raw_current = Rotation.from_matrix(raw_delta) * hand_ref
        # q and -q must describe exactly the same pose.
        controller["orientation"] = (-raw_current.as_quat()).tolist()
        # Feedback can lag and change without changing the reference target.
        measured[index + 3 : index + 6] += [0.02, -0.01, 0.03]
        arm._expert._last_update_time = time.time()
        _, info = map_state(station, measured)
        assert info["yam_pico_fault"] is None
        target = station.kinematics[hand].targets[-1]
        np.testing.assert_allclose(target[:3, 3], reference[:3, 3], atol=1e-7)
        np.testing.assert_allclose(target[:3, :3], delta @ reference[:3, :3], atol=1e-7)


def test_operator_rotation_regrip_recaptures_reference_and_scales_angle(monkeypatch):
    from rlinf.robotics.parts.teleop.yam_pico import _delta_to_tcp_pose, _YamPicoArm
    from rlinf.robotics.parts.transports.pico import PicoExpert

    monkeypatch.setattr(PicoExpert, "start", lambda self: None)
    arm = _YamPicoArm(
        hand="left",
        rotation_delta_frame="operator",
        rotation_scale=0.5,
        calibration={"enabled": False},
    )
    controller = {"position": [0, 1, 0], "orientation": [0, 0, 0, 1], "grip": 1.0}
    arm._expert._latest_data = {"left_controller": controller}
    arm._expert._last_update_time = time.time()
    tcp = np.r_[0.2, 0.1, 0.3, Rotation.from_euler("xyz", [0.2, -0.3, 0.4]).as_quat()]
    scale = np.array([0.005, 0.03, 1.0])
    try:
        arm.command(arm.read(), tcp, scale, clip_motion=False)
        controller["orientation"] = (
            Rotation.from_euler("y", 20, degrees=True).as_quat().tolist()
        )
        action, _, _ = arm.command(arm.read(), tcp, scale, clip_motion=False)
        target = _delta_to_tcp_pose(action, tcp, scale, clip_motion=False)
        expected = Rotation.from_euler("z", 10, degrees=True) * Rotation.from_quat(
            tcp[3:]
        )
        np.testing.assert_allclose(
            Rotation.from_quat(target[3:]).as_matrix(), expected.as_matrix(), atol=1e-7
        )
        controller["grip"] = 0.0
        arm.command(arm.read(), target, scale, clip_motion=False)
        controller["orientation"] = (
            Rotation.from_euler("xyz", [40, -30, 20], degrees=True).as_quat().tolist()
        )
        controller["grip"] = 1.0
        action, _, _ = arm.command(arm.read(), target, scale, clip_motion=False)
        np.testing.assert_allclose(action[:6], 0.0, atol=1e-6)
    finally:
        arm.stop()


@pytest.mark.parametrize("hand", ["left", "right"])
@pytest.mark.parametrize("limiter", ["wrist", "arm"])
def test_per_joint_bounds_preserve_coordinated_motion(monkeypatch, hand, limiter):
    station = build_station(max_joint_delta_per_joint=[0.08] * 3 + [0.12] * 3)
    try:
        station.env.reset()
        station.env.step(np.zeros(14))
        station.arms[hand].held = True
        station.env.step(np.zeros(14))
        before = station.base.get_hold_action().copy()
        index = 0 if hand == "left" else 7
        # For the second case J1 determines the fraction even though J5 moves more.
        dq = np.array([0.04, 0.02, 0.01, 0.10, -0.24, 0.06])
        if limiter == "arm":
            dq[0] = 0.2
        expected_fraction = 0.5 if limiter == "wrist" else 0.4
        expected_joint = 5 if limiter == "wrist" else 1
        monkeypatch.setattr(
            station.kinematics[hand],
            "solve",
            lambda target, seed, gripper: YamIKResult(True, seed + dq, 0, 0, 0),
        )
        info = station.env.step(np.zeros(14))[-1]
        assert info["yam_pico_fault"] is None
        assert info[f"{hand}_joint_step_fraction"] == pytest.approx(expected_fraction)
        assert info[f"{hand}_limiting_joint"] == expected_joint
        np.testing.assert_allclose(
            info["accepted_action"][index : index + 6],
            before[index : index + 6] + expected_fraction * dq,
            atol=1e-7,
        )
        np.testing.assert_array_equal(info["intervene_action"], info["accepted_action"])
        # The downstream runtime must not clip wrist steps back to the scalar 0.08.
        assert (
            np.max(
                np.abs(
                    info["accepted_action"][index + 3 : index + 6]
                    - before[index + 3 : index + 6]
                )
            )
            > 0.08
        )
        station.arms[hand].held = False
        inactive = station.env.step(np.zeros(14))[-1]
        assert inactive[f"{hand}_joint_step_fraction"] == 1.0
        assert inactive[f"{hand}_limiting_joint"] == 0
    finally:
        station.env.close()


def test_timing_separates_camera_wait_from_time_outside_control(
    station, monkeypatch, caplog
):
    import rlinf.envs.real.yam.dual_yam_joint_env as env_module

    base = station.base
    frames = base._get_observation()["frames"]
    base.config.is_dummy = False  # Hardware is already connected to mock backends.
    now = [100.0]
    monkeypatch.setattr(env_module.time, "monotonic", lambda: now[0])
    base._last_dispatch_time_s = None
    station.env.config.log_control_timing = True

    def delayed_frames():
        now[0] += 0.08
        return frames

    monkeypatch.setattr(base, "_read_camera_frames", delayed_frames)
    held = base.get_hold_action().copy()
    first = station.env.step(np.zeros(14))[-1]
    assert first["yam_camera_read_s"] == pytest.approx(0.08)
    assert first["yam_command_interval_s"] == 0.0
    now[0] += 0.42
    second = station.env.step(np.zeros(14))[-1]
    assert second["yam_command_interval_s"] == pytest.approx(0.50)
    assert second["yam_between_ticks_s"] == pytest.approx(0.42)
    assert second["yam_tick_s"] == pytest.approx(0.08)
    np.testing.assert_array_equal(second["accepted_action"], held)
    now[0] += 0.42
    with caplog.at_level("INFO"):
        station.env.step(np.zeros(14))
    assert "YAM VR timing:" in caplog.text
    assert "gap=500.0" in caplog.text
    assert "cameras=80.0" in caplog.text
    assert "outside=420.0" in caplog.text
    assert "fault_ticks=0" in caplog.text
