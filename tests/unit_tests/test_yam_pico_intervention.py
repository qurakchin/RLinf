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

"""YAM VR ownership, recording and accepted-action contracts with mock hardware."""

import time

import gymnasium as gym
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rlinf.envs.real.yam.config import YamPicoConfig
from rlinf.envs.real.yam.dual_yam_joint_env import DualYamJointEnv
from rlinf.envs.real.yam.kinematics import YamIKResult
from rlinf.envs.real.yam.pico_intervention import DualYamPicoIntervention


class Expert:
    def __init__(self):
        self.ready = True
        self.active = False
        self.buttons = {}
        self.action = np.array([0.5, 0, 0, 0, 0, 0, 0.0])
        self.reference = False
        self.resets = 0
        self.stopped = False

    def get_action(self, tcp_pose, action_scale, **kwargs):
        action = self.action.copy()
        if self.active and not self.reference:
            action[:6] = 0
        self.reference = self.active
        return (
            action,
            self.active and self.ready,
            {
                "pico_ready": self.ready,
                "pico_active": self.active,
                "pico_calibrated": True,
                "pico_control_value": float(self.active),
            },
        )

    def reset_reference(self):
        self.reference = False
        self.resets += 1

    def read_buttons(self):
        return self.buttons if self.ready else {}

    def stop(self):
        self.stopped = True


class Kinematics:
    def __init__(self):
        self.fail = False
        self.targets = []

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


@pytest.fixture
def env(monkeypatch):
    base = DualYamJointEnv(
        {
            "is_dummy": True,
            "image_height": 8,
            "image_width": 8,
            "manual_episode_control_only": True,
        }
    )
    monkeypatch.setattr(base, "_pace", lambda: None)
    monkeypatch.setattr(
        base.runtime, "connect_leaders", lambda: pytest.fail("VR opened leader CAN")
    )
    wrapper = DualYamPicoIntervention(
        base,
        {"wait_for_record_button": False, "button_debounce_s": 0.0},
        experts={s: Expert() for s in ("left", "right")},
        kinematics={s: Kinematics() for s in ("left", "right")},
    )
    wrapper.reset()
    wrapper.step(np.zeros(14))  # Observe released grips, prime button edges.
    yield wrapper
    wrapper.close()


def press(env, button):
    env.experts["right"].buttons = {}
    env.step(np.zeros(14))
    env.experts["right"].buttons = {button: True}
    return env.step(np.zeros(14))


def test_single_arm_first_engagement_and_gripper_latch(env):
    expert = env.experts["left"]
    expert.active = True
    _, _, _, _, first = env.step(np.ones(14))
    np.testing.assert_allclose(first["intervene_action"][:6], 0.0)
    expert.action[6] = 1.0
    _, _, _, _, info = env.step(np.ones(14))
    assert info["intervene_action"][0] > 0
    assert info["intervene_action"][6] == 1.0
    np.testing.assert_array_equal(info["intervene_action"][7:], np.zeros(7))
    expert.action[6] = 0.0
    assert env.step(np.zeros(14))[-1]["intervene_action"][6] == 1.0
    expert.active = False
    held = env.unwrapped.get_hold_action()
    np.testing.assert_allclose(env.step(np.zeros(14))[-1]["intervene_action"], held)


@pytest.mark.parametrize("side,index", [("left", 0), ("right", 7)])
def test_inactive_arm_holds_fixed_joints_and_recaptures_release_pose(
    env, monkeypatch, side, index
):
    measured = env.unwrapped.get_hold_action().copy()
    initial = measured.copy()
    joints = slice(index, index + 6)
    monkeypatch.setattr(env.unwrapped, "get_hold_action", lambda: measured.copy())

    # Startup idle keeps its reference even if the encoder reports drift.
    measured[joints] += 0.02
    info = env.step(np.zeros(14))[-1]
    np.testing.assert_allclose(info["intervene_action"][joints], initial[joints])
    assert not info["pico_active"]

    # Taking control uses the current pose, not the old idle target.
    env.experts[side].active = True
    info = env.step(np.zeros(14))[-1]
    np.testing.assert_allclose(info["intervene_action"][joints], measured[joints])
    assert info[side]

    # Releasing the grip captures a new reference once for this arm only.
    measured[joints] += 0.02
    released = measured.copy()
    env.experts[side].active = False
    env.step(np.zeros(14))
    for _ in range(3):
        measured[joints] += 0.01
        info = env.step(np.zeros(14))[-1]
        np.testing.assert_allclose(
            info["intervene_action"][joints], released[joints]
        )
        other_joints = slice(7, 13) if index == 0 else slice(0, 6)
        np.testing.assert_allclose(
            info["intervene_action"][other_joints], initial[other_joints]
        )


def test_data_fault_recovery_replaces_old_idle_reference(env, monkeypatch):
    measured = env.unwrapped.get_hold_action().copy()
    monkeypatch.setattr(env.unwrapped, "get_hold_action", lambda: measured.copy())
    measured[:6] += 0.02
    env.experts["right"].ready = False
    info = env.step(np.zeros(14))[-1]
    assert info["yam_pico_fault"]
    np.testing.assert_allclose(info["intervene_action"], measured)

    # Recovery captures the actual pose instead of restoring a pre-fault target.
    measured[:6] += 0.02
    recovered = measured.copy()
    env.experts["right"].ready = True
    assert env.step(np.zeros(14))[-1]["yam_pico_fault"] is None
    measured[:6] += 0.02
    info = env.step(np.zeros(14))[-1]
    np.testing.assert_allclose(info["intervene_action"][:6], recovered[:6])


@pytest.mark.parametrize("fault", ["stale", "nan"])
def test_fault_holds_both_and_requires_release_before_reengagement(env, fault):
    for e in env.experts.values():
        e.active = True
    env.step(np.zeros(14))
    held = env.unwrapped.get_hold_action()
    if fault == "stale":
        env.experts["right"].ready = False
    else:
        env.experts["right"].action[0] = np.nan
    info = env.step(np.zeros(14))[-1]
    assert info["yam_pico_fault"]
    np.testing.assert_allclose(info["intervene_action"], held)
    env.experts["right"].ready = True
    env.experts["right"].action[0] = 0.5
    assert env.step(np.zeros(14))[-1]["yam_pico_fault"]
    for e in env.experts.values():
        e.active = False
    assert env.step(np.zeros(14))[-1]["yam_pico_fault"] is None
    env.experts["left"].active = True
    np.testing.assert_allclose(env.step(np.zeros(14))[-1]["intervene_action"], held)


@pytest.mark.parametrize("side", ["left", "right"])
def test_ik_failure_holds_only_failed_arm_and_retries_without_release(env, side):
    for expert in env.experts.values():
        expert.active = True
    env.step(np.zeros(14))
    press(env, "right_menu_button")
    env.experts["right"].buttons = {}
    held = env.unwrapped.get_hold_action().copy()
    resets = {name: expert.resets for name, expert in env.experts.items()}
    for expert in env.experts.values():
        expert.action[6] = 1.0
    kin = env.kinematics[side]
    failed_index = 0 if side == "left" else 7
    healthy_index = 7 - failed_index
    healthy_side = "right" if side == "left" else "left"
    kin.fail = True
    solves = len(kin.targets)
    for attempt in range(3):
        _, reward, done, _, info = env.step(np.zeros(14))
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
        for name, expert in env.experts.items():
            assert expert.active and expert.reference
            assert expert.resets == resets[name]

    kin.fail = False
    for expert in env.experts.values():
        expert.action[6] = 0.0
    info = env.step(np.zeros(14))[-1]
    assert info["yam_pico_fault"] is None and info["pico_active"]
    assert info["intervene_action"][0] > held[0]
    assert info["intervene_action"][7] > held[7]
    # An open command from a rejected frame must not leak into the retry.
    assert info["intervene_action"][failed_index + 6] == held[failed_index + 6]
    assert info["intervene_action"][healthy_index + 6] == 1.0


def test_both_ik_failures_hold_both_and_report_each_reason(env):
    for expert in env.experts.values():
        expert.active = True
    env.step(np.zeros(14))
    held = env.unwrapped.get_hold_action().copy()
    for kin in env.kinematics.values():
        kin.fail = True
    info = env.step(np.zeros(14))[-1]
    np.testing.assert_allclose(info["intervene_action"], held)
    assert not info["pico_active"]
    assert info["left_ik_fault"] == info["right_ik_fault"] == "injected_failure"


def test_data_loss_during_ik_retry_still_requires_grip_release(env):
    env.experts["left"].active = True
    env.step(np.zeros(14))
    env.kinematics["left"].fail = True
    assert env.step(np.zeros(14))[-1]["yam_pico_fault"].startswith("left:ik:")
    env.experts["right"].ready = False
    assert env.step(np.zeros(14))[-1]["yam_pico_fault"] == "right:unavailable"
    env.kinematics["left"].fail = False
    env.experts["right"].ready = True
    assert env.step(np.zeros(14))[-1]["yam_pico_fault"] == "right:unavailable"
    env.experts["left"].active = False
    assert env.step(np.zeros(14))[-1]["yam_pico_fault"] is None


def test_record_edges_abort_and_success_preserve_only_manual_boundary(env):
    env.experts["left"].active = True
    info = press(env, "right_menu_button")[-1]
    assert info["record_reset"] and not info["pre_record"]
    assert not env.step(np.zeros(14))[2]  # Held button cannot end again.
    result = press(env, "right_menu_button")
    assert result[1] == 1 and result[2] and result[-1]["success"]
    count = env.experts["left"].resets
    env.reset()
    assert env.experts["left"].resets == count
    env.reset()  # Early reset must clear the reference.
    assert env.experts["left"].resets > count
    env.experts["left"].active = False
    env.step(np.zeros(14))
    press(env, "right_menu_button")
    info = press(env, "left_menu_button")[-1]
    assert info["pre_record"] and info["record_reset"] and not info["success"]


def test_fault_discards_active_recording(env):
    press(env, "right_menu_button")
    env.experts["left"].ready = False
    _, reward, done, _, info = env.step(np.zeros(14))
    assert info["record_reset"] and info["pre_record"]
    assert not done and reward == 0.0


def test_recorded_action_uses_runtime_acceptance(env, monkeypatch):
    original = env.env.step

    def clipped(action):
        obs, reward, done, truncated, info = original(action)
        info["accepted_action"] = np.full(14, 0.123, dtype=np.float32)
        return obs, reward, done, truncated, info

    monkeypatch.setattr(env.env, "step", clipped)
    np.testing.assert_allclose(env.step(np.zeros(14))[-1]["intervene_action"], 0.123)


@pytest.mark.parametrize("joint_step", [0.05, 0.08])
def test_large_valid_ik_target_is_interpolated_and_recorded(
    env, monkeypatch, joint_step
):
    env.unwrapped.config.max_joint_delta = joint_step
    env.experts["left"].active = True
    env.step(np.zeros(14))
    before = env.unwrapped.get_hold_action().copy()

    def large_target(target, seed, gripper):
        return YamIKResult(
            True, seed + np.array([0.2, 0.1, 0.05, 0.1, 0, -0.1]), 0.0, 0.0, 0.0
        )

    monkeypatch.setattr(env.kinematics["left"], "solve", large_target)
    info = env.step(np.zeros(14))[-1]
    assert info["yam_pico_fault"] is None and info["pico_active"]
    np.testing.assert_allclose(
        info["intervene_action"][:6],
        before[:6] + joint_step * np.array([1, 0.5, 0.25, 0.5, 0, -0.5]),
    )
    np.testing.assert_allclose(info["intervene_action"][6:], before[6:])
    np.testing.assert_array_equal(info["intervene_action"], info["accepted_action"])


def test_reset_wait_keeps_teleop_ticking_without_episode_steps(env):
    env.config.wait_for_record_button = True

    class Keyboard:
        count = 0

        def poll(self):
            self.count += 1
            return {"record"} if self.count == 3 else set()

        def close(self):
            pass

    env._keyboard = Keyboard()
    _, info = env.reset()
    assert env._keyboard.count == 3
    assert not info["pre_record"]
    assert env.unwrapped._num_steps == 0


def test_factory_rejects_unsafe_combinations_before_env_construction(monkeypatch):
    from rlinf.envs.real.yam import tasks

    monkeypatch.setattr(
        tasks, "DualYamJointEnv", lambda **kw: pytest.fail("constructed env")
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        tasks.create_dual_yam_joint_env(
            {"leader_intervention": {"enabled": True}},
            None,
            None,
            0,
            {"use_pico": True},
        )
    with pytest.raises(ValueError, match="enforce_runtime"):
        tasks.create_dual_yam_joint_env(
            {"enforce_runtime_joint_limits": False},
            None,
            None,
            0,
            {"use_pico": True},
        )


def test_real_pico_arm_lifecycle_preserves_calibration(monkeypatch):
    from rlinf.envs.real.yam.pico_intervention import _YamPicoArm
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
    action, replaced, _ = arm.get_action(tcp, np.array([0.01, 0.1, 1.0]))
    assert replaced
    np.testing.assert_allclose(action[:6], 0)
    assert arm.read_buttons()["A"]
    arm._expert._calibrated = True
    arm.reset_reference()
    assert arm._expert._calibrated and arm._ref_tcp_pos is None
    action, _, _ = arm.get_action(tcp, np.array([0.01, 0.1, 1.0]))
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
    env, monkeypatch, side, success_attempt
):
    env.config.ik_backtrack_attempts = 3
    env.config.ik.max_solve_s = 1.0
    arm = np.array([0.1, 0.2, 0.3, 0.4, -0.3, 0.2, 0.5])
    original_arm = arm.copy()
    current = env.kinematics[side].fk(arm[:6], arm[6])
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

    monkeypatch.setattr(env.kinematics[side], "solve", solve)
    result, attempts, fraction, first_reason = env._solve_with_backtracking(
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
    env, monkeypatch, reason, retries
):
    env.config.ik_backtrack_attempts = 3
    env.config.ik.max_solve_s = 1.0
    monkeypatch.setattr(
        env.kinematics["left"],
        "solve",
        lambda *args: YamIKResult(False, np.zeros(6), 0, 0, 0, reason),
    )
    result, attempts, _, _ = env._solve_with_backtracking(
        "left", np.eye(4), np.eye(4), np.zeros(7)
    )
    assert not result.success
    assert attempts == 1 + retries
    assert result.reason == reason


@pytest.mark.parametrize("durations", [[0.04], [0.01, 0.021]])
def test_ik_backtracking_shares_budget_and_rejects_late_success(
    env, monkeypatch, durations
):
    import rlinf.envs.real.yam.pico_intervention as module

    env.config.ik_backtrack_attempts = 3
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

    monkeypatch.setattr(env.kinematics["left"], "solve", solve)
    result, attempts, _, _ = env._solve_with_backtracking(
        "left", np.eye(4), np.eye(4), np.zeros(7)
    )
    assert attempts == len(durations)
    assert not result.success
    assert result.reason == "solve_timeout"
    assert result.elapsed_s == pytest.approx(sum(durations))


@pytest.mark.parametrize("side,index", [("left", 0), ("right", 7)])
@pytest.mark.parametrize("recover", [True, False])
def test_ik_backtracking_dispatches_only_valid_reduced_targets(
    env, monkeypatch, side, index, recover
):
    env.config.ik_backtrack_attempts = 3
    env.config.ik.max_solve_s = 1.0
    for expert in env.experts.values():
        expert.active = True
    env.step(np.zeros(14))
    before = env.unwrapped.get_hold_action().copy()
    env.experts[side].action[0] = 4.0  # Full target is 20 mm ahead.
    original_solve = env.kinematics[side].solve
    calls = []

    def solve(target, seed, gripper):
        calls.append(target.copy())
        if recover and len(calls) == 3:
            return original_solve(target, seed, gripper)
        return YamIKResult(False, np.full(6, 99.0), 1, 1, 0, "not_converged")

    monkeypatch.setattr(env.kinematics[side], "solve", solve)
    info = env.step(np.zeros(14))[-1]
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
    assert env._fault is None  # Exhaustion does not latch both arms.
    assert env.experts[side].reference


def test_two_real_subscribers_receive_and_close_independently(tmp_path):
    zmq = pytest.importorskip("zmq")
    from rlinf.robotics.parts.transports.pico import PicoExpert

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    address = f"ipc://{tmp_path}/pico.ipc"
    publisher.bind(address)
    experts = []
    try:
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
        publisher.close(linger=0)
        context.term()
    assert all(not thread.is_alive() for thread in threads)


def test_rotation_delta_uses_base_frame_left_multiplication(env):
    expert = env.experts["left"]
    expert.active = True
    env.step(np.zeros(14))
    expert.action[:] = [0, 0, 0, 1, 0, 0, 0]
    before = env.unwrapped.get_hold_action()[:6]
    env.step(np.zeros(14))
    expected = (
        Rotation.from_rotvec([env.config.max_rotation_delta, 0, 0]).as_matrix()
        @ Rotation.from_rotvec(before[3:6]).as_matrix()
    )
    np.testing.assert_allclose(env.kinematics["left"].targets[-1][:3, :3], expected)


def test_runtime_rejection_aborts_recording_and_reports_accepted_hold(env, monkeypatch):
    press(env, "right_menu_button")
    original = env.env.step

    def reject(action):
        result = original(env.unwrapped.get_hold_action())
        result[-1]["action_rejected"] = "measured_joint_out_of_limits"
        return result

    monkeypatch.setattr(env.env, "step", reject)
    info = env.step(np.zeros(14))[-1]
    assert info["record_reset"] and info["pre_record"]
    assert info["yam_pico_fault"].startswith("runtime:")
    np.testing.assert_array_equal(info["accepted_action"], info["intervene_action"])


def test_experts_are_stopped_when_model_initialization_fails(monkeypatch):
    from rlinf.envs.real.yam import pico_intervention

    experts = {s: Expert() for s in ("left", "right")}
    base = DualYamJointEnv({"is_dummy": True})

    def fail(**kwargs):
        raise ValueError("invalid model")

    monkeypatch.setattr(pico_intervention, "YamKinematicsAdapter", fail)
    wrapper = DualYamPicoIntervention(base, experts=experts)
    with pytest.raises(ValueError, match="invalid model"):
        wrapper.reset()
    assert all(e.stopped for e in experts.values())
    assert base._closed


def test_unexpected_control_exception_closes_followers_and_experts(env, monkeypatch):
    def fail(*args):
        raise RuntimeError("injected FK error")

    monkeypatch.setattr(env.kinematics["left"], "fk", fail)
    with pytest.raises(RuntimeError, match="injected FK error"):
        env.step(np.zeros(14))
    assert env.unwrapped._closed
    assert all(e.stopped for e in env.experts.values())


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
    env, tmp_path, monkeypatch, write_to_disk
):
    from rlinf.envs.wrappers.collect_episode import CollectEpisode

    collector = CollectEpisode(
        PackedObservations(env),
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
        env.experts["left"].active = False
        collector.step(np.zeros(14))

        def edge(button):
            env.experts["right"].buttons = {}
            collector.step(np.zeros(14))
            env.experts["right"].buttons = {button: True}
            return collector.step(np.zeros(14))

        edge("right_menu_button")
        collector.step(np.zeros(14))
        edge("left_menu_button")
        assert collector._buffers[0]["actions"] == []
        for _ in range(2):
            edge("right_menu_button")
            before = env.unwrapped.get_hold_action().copy()
            env.experts["left"].active = True
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
    env, monkeypatch, hand, initial_yaw_deg, motion
):
    """The operator's motion adds to where the arm stood at the grip edge.

    A target composed from the *current* feedback instead would stall while the
    position controller catches up, which is what ``motion=0.2`` reproduces.
    """
    from rlinf.envs.real.yam.pico_intervention import _YamPicoArm
    from rlinf.robotics.parts.transports.pico import PicoExpert

    monkeypatch.setattr(PicoExpert, "start", lambda self: None)
    arm = _YamPicoArm(**env.config.expert_kwargs(hand))
    env.experts[hand] = arm
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
                q = env.unwrapped.get_hold_action()[index : index + 7]
                anchor = env.kinematics[hand].fk(q[:6], q[6])[:3, 3].copy()
            _, _, _, _, info = env.step(np.zeros(14))
            assert info["yam_pico_fault"] is None
            target = env.kinematics[hand].targets[-1]
            np.testing.assert_allclose(
                target[:3, 3], anchor + reading["position_delta"], atol=1e-7
            )
    finally:
        arm.stop()


def test_composed_tcp_target_preserves_the_original_equations():
    from rlinf.envs.real.yam.pico_intervention import _delta_to_tcp_pose

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
    env, monkeypatch, initial_yaw_deg
):
    from rlinf.envs.real.yam.pico_intervention import _YamPicoArm
    from rlinf.robotics.parts.transports.pico import R_PICO_TO_WORLD, PicoExpert

    monkeypatch.setattr(PicoExpert, "start", lambda self: None)
    kwargs = env.config.expert_kwargs("right")
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
    arm.get_action(tcp, scale)
    controller["position"] = (R_PICO_TO_WORLD.T @ np.array([0.05, 0.005, 0])).tolist()
    current = Rotation.from_rotvec([0.01, 0, 0]) * ref
    controller["orientation"] = current.as_quat().tolist()
    action, _, _ = arm.get_action(tcp, scale)
    np.testing.assert_allclose(action[:3], [1, 1, 0])
    local_delta = (ref.inv() * current).as_rotvec()
    expected = np.array([-local_delta[2], -local_delta[0], local_delta[1]])
    np.testing.assert_allclose(action[3:6] * scale[1], expected, atol=1e-8)
    arm.stop()


@pytest.mark.parametrize("hand", ["left", "right"])
def test_full_vr_lift_target_survives_stalled_feedback(env, monkeypatch, hand):
    from rlinf.envs.real.yam.pico_intervention import _YamPicoArm
    from rlinf.robotics.parts.transports.pico import PicoExpert

    monkeypatch.setattr(PicoExpert, "start", lambda self: None)
    env.config.position_scale = 1.0
    arm = _YamPicoArm(**env.config.expert_kwargs(hand))
    env.experts[hand] = arm
    measured = env.unwrapped.get_hold_action().copy()
    monkeypatch.setattr(env.unwrapped, "get_hold_action", lambda: measured.copy())
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
    env._compute_action()  # Grip captures the measured TCP reference.
    start = env.kinematics[hand].targets[-1].copy()
    controller["position"][1] += 0.15  # Raw VR +Y maps to arm +Z.
    for _ in range(3):
        arm._expert._last_update_time = time.time()
        _, info = env._compute_action()
        assert info["yam_pico_fault"] is None
        target = env.kinematics[hand].targets[-1]
        np.testing.assert_allclose(
            target[:3, 3] - start[:3, 3], [0, 0, 0.15], atol=1e-7
        )


@pytest.mark.parametrize("hand", ["left", "right"])
@pytest.mark.parametrize("initial_hand_deg", [[0, 0, 0], [0, 0, 90], [35, -25, 70]])
@pytest.mark.parametrize("head_yaw_deg,robot_yaw", [(0, 0), (90, 0.4)])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_yam_spatial_rotation_matches_translation_axes(
    env, monkeypatch, hand, initial_hand_deg, head_yaw_deg, robot_yaw, axis
):
    from rlinf.envs.real.yam.pico_intervention import _YamPicoArm
    from rlinf.robotics.parts.transports.pico import R_PICO_TO_WORLD, PicoExpert

    monkeypatch.setattr(PicoExpert, "start", lambda self: None)
    env.config.rotation_scale = 1.0
    getattr(env.config, hand)["operator_to_robot_yaw"] = robot_yaw
    arm = _YamPicoArm(**env.config.expert_kwargs(hand))
    env.experts[hand] = arm
    index = 0 if hand == "left" else 7
    measured = env.unwrapped.get_hold_action().copy()
    measured[index : index + 6] = [0.2, 0.3, 0.4, 0.3, -0.4, 0.5]
    monkeypatch.setattr(env.unwrapped, "get_hold_action", lambda: measured.copy())
    reference = env.kinematics[hand].fk(
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
    env._compute_action()
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
        _, info = env._compute_action()
        assert info["yam_pico_fault"] is None
        target = env.kinematics[hand].targets[-1]
        np.testing.assert_allclose(target[:3, 3], reference[:3, 3], atol=1e-7)
        np.testing.assert_allclose(target[:3, :3], delta @ reference[:3, :3], atol=1e-7)


def test_operator_rotation_regrip_recaptures_reference_and_scales_angle(monkeypatch):
    from rlinf.envs.real.yam.pico_intervention import (
        _delta_to_tcp_pose,
        _YamPicoArm,
    )
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
        arm.get_action(tcp, scale, clip_motion=False)
        controller["orientation"] = (
            Rotation.from_euler("y", 20, degrees=True).as_quat().tolist()
        )
        action, _, _ = arm.get_action(tcp, scale, clip_motion=False)
        target = _delta_to_tcp_pose(action, tcp, scale, clip_motion=False)
        expected = Rotation.from_euler("z", 10, degrees=True) * Rotation.from_quat(
            tcp[3:]
        )
        np.testing.assert_allclose(
            Rotation.from_quat(target[3:]).as_matrix(), expected.as_matrix(), atol=1e-7
        )
        controller["grip"] = 0.0
        arm.get_action(target, scale, clip_motion=False)
        controller["orientation"] = (
            Rotation.from_euler("xyz", [40, -30, 20], degrees=True).as_quat().tolist()
        )
        controller["grip"] = 1.0
        action, _, _ = arm.get_action(target, scale, clip_motion=False)
        np.testing.assert_allclose(action[:6], 0.0, atol=1e-6)
    finally:
        arm.stop()


@pytest.mark.parametrize("hand", ["left", "right"])
@pytest.mark.parametrize("limiter", ["wrist", "arm"])
def test_per_joint_bounds_preserve_coordinated_motion(env, monkeypatch, hand, limiter):
    env.unwrapped.config.max_joint_delta_per_joint = [0.08] * 3 + [0.12] * 3
    env.experts[hand].active = True
    env.step(np.zeros(14))
    before = env.unwrapped.get_hold_action().copy()
    index = 0 if hand == "left" else 7
    # For the second case J1 determines the fraction even though J5 moves more.
    dq = np.array([0.04, 0.02, 0.01, 0.10, -0.24, 0.06])
    if limiter == "arm":
        dq[0] = 0.2
    expected_fraction = 0.5 if limiter == "wrist" else 0.4
    expected_joint = 5 if limiter == "wrist" else 1
    monkeypatch.setattr(
        env.kinematics[hand],
        "solve",
        lambda target, seed, gripper: YamIKResult(True, seed + dq, 0, 0, 0),
    )
    info = env.step(np.zeros(14))[-1]
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
    env.experts[hand].active = False
    inactive = env.step(np.zeros(14))[-1]
    assert inactive[f"{hand}_joint_step_fraction"] == 1.0
    assert inactive[f"{hand}_limiting_joint"] == 0


def test_timing_separates_camera_wait_from_time_outside_control(
    env, monkeypatch, caplog
):
    import rlinf.envs.real.yam.dual_yam_joint_env as env_module

    base = env.unwrapped
    frames = base._get_observation()["frames"]
    base.config.is_dummy = False  # Hardware is already connected to mock backends.
    now = [100.0]
    monkeypatch.setattr(env_module.time, "monotonic", lambda: now[0])
    base._last_dispatch_time_s = None
    env.config.log_control_timing = True

    def delayed_frames():
        now[0] += 0.08
        return frames

    monkeypatch.setattr(base, "_read_camera_frames", delayed_frames)
    held = base.get_hold_action().copy()
    first = env.step(np.zeros(14))[-1]
    assert first["yam_camera_read_s"] == pytest.approx(0.08)
    assert first["yam_pico_compute_s"] == pytest.approx(0.0)
    assert first["yam_command_interval_s"] == 0.0
    now[0] += 0.42
    second = env.step(np.zeros(14))[-1]
    assert second["yam_command_interval_s"] == pytest.approx(0.50)
    assert second["yam_between_ticks_s"] == pytest.approx(0.42)
    assert second["yam_tick_s"] == pytest.approx(0.08)
    np.testing.assert_array_equal(second["accepted_action"], held)
    now[0] += 0.42
    with caplog.at_level("INFO"):
        env.step(np.zeros(14))
    assert "YAM VR timing:" in caplog.text
    assert "gap=500.0" in caplog.text
    assert "cameras=80.0" in caplog.text
    assert "outside=420.0" in caplog.text
    assert "fault_ticks=0" in caplog.text
