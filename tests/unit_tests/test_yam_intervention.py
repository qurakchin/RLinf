# Copyright 2026 The RLinf Authors.
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

"""Policy/leader ownership tests for the dual-YAM intervention wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from rlinf.envs.real.yam.dual_yam_joint_env import DualYamJointEnv
from rlinf.envs.real.yam.leader_intervention import (
    DualYamLeaderIntervention,
)
from rlinf.envs.real.yam.types import (
    DualYamState,
    YamArmState,
    YamCommandResult,
    split_dual_action,
)


class _Runtime:
    def __init__(self) -> None:
        self.state = np.zeros(14, dtype=np.float64)
        self.leader_action = np.zeros(14, dtype=np.float64)
        self.buttons = (False, False)
        self.commands: list[np.ndarray] = []
        self.leader_commands: list[np.ndarray] = []
        self.moves: list[np.ndarray] = []
        self.hold_calls = 0
        self.emergency_hold_calls = 0
        self.release_calls = 0
        self.engage_calls = 0
        self.close_calls = 0
        self.fail_hold = False

    def connect_followers(self) -> None:
        return None

    def connect_leaders(self) -> None:
        return None

    def read_state(self) -> DualYamState:
        left, right = split_dual_action(self.state)
        return DualYamState(
            left=YamArmState(left[:6], left[6], 1.0),
            right=YamArmState(right[:6], right[6], 1.0),
        )

    def read_leader_action(self) -> tuple[np.ndarray, tuple[bool, bool]]:
        return self.leader_action.copy(), self.buttons

    def command(self, action: np.ndarray) -> YamCommandResult:
        requested = np.asarray(action, dtype=np.float64).copy()
        self.commands.append(requested)
        self.state = requested
        return YamCommandResult(requested=requested, accepted=requested)

    def move_to(self, target: np.ndarray, **_kwargs) -> np.ndarray:
        target = np.asarray(target, dtype=np.float64).copy()
        self.moves.append(target)
        self.state = target
        return target.copy()

    def validate_leader_target(self, target: np.ndarray) -> None:
        assert np.isfinite(target).all()

    def command_leaders(self, target: np.ndarray) -> None:
        self.leader_commands.append(target.copy())
        joints = [*range(6), *range(7, 13)]
        self.leader_action[joints] = target[joints]

    def hold(self) -> np.ndarray:
        self.hold_calls += 1
        if self.fail_hold:
            raise RuntimeError("hold failed")
        return self.state.copy()

    def emergency_hold(self) -> None:
        self.emergency_hold_calls += 1

    def engage(self) -> YamCommandResult:
        self.engage_calls += 1
        self.state = self.leader_action.copy()
        return YamCommandResult(
            requested=self.leader_action.copy(),
            accepted=self.leader_action.copy(),
        )

    def apply_leader_feedback(self) -> None:
        return None

    def release_leader_feedback(self) -> None:
        self.release_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def test_policy_passthrough_and_leader_takeover_have_distinct_info(monkeypatch):
    runtime = _Runtime()
    base_env = DualYamJointEnv(
        override_cfg={"is_dummy": True, "dummy_camera_names": ["top_rgb"]},
        runtime=runtime,
    )
    monkeypatch.setattr(base_env, "_pace", lambda: None)
    env = DualYamLeaderIntervention(
        base_env,
        {
            "wait_for_record_button": False,
            "button_debounce_s": 0.0,
            "unsynced_action_source": "policy",
        },
    )
    env.reset()
    policy_action = np.linspace(0.0, 0.13, 14)

    _, _, _, _, policy_info = env.step(policy_action)

    np.testing.assert_allclose(runtime.commands[-1], policy_action)
    assert "intervene_action" not in policy_info
    assert not policy_info["intervened"]

    runtime.buttons = (True, False)
    runtime.leader_action = np.linspace(0.2, 0.33, 14)
    _, _, _, _, leader_info = env.step(policy_action)

    assert runtime.engage_calls == 1
    np.testing.assert_allclose(runtime.commands[-1], runtime.leader_action)
    np.testing.assert_allclose(
        leader_info["intervene_action"], runtime.leader_action.astype(np.float32)
    )
    assert leader_info["intervened"]

    runtime.buttons = (False, False)
    env.step(policy_action)
    runtime.buttons = (True, False)
    env.step(policy_action)

    assert runtime.release_calls == 2  # reset plus sync-off
    assert not env._sync_enabled


def test_sync_off_failure_still_releases_feedback_and_clears_ownership(monkeypatch):
    runtime = _Runtime()
    base_env = DualYamJointEnv(
        override_cfg={"is_dummy": True, "dummy_camera_names": ["top_rgb"]},
        runtime=runtime,
    )
    monkeypatch.setattr(base_env, "_pace", lambda: None)
    env = DualYamLeaderIntervention(
        base_env,
        {
            "wait_for_record_button": False,
            "button_debounce_s": 0.0,
            "unsynced_action_source": "policy",
        },
    )
    env.reset()

    runtime.buttons = (True, False)
    env.step(np.zeros(14))
    assert env._sync_enabled

    runtime.buttons = (False, False)
    env.step(np.zeros(14))
    runtime.buttons = (True, False)
    runtime.fail_hold = True

    with pytest.raises(RuntimeError, match="hold failed"):
        env.step(np.zeros(14))

    assert not env._sync_enabled
    assert runtime.emergency_hold_calls == 1
    assert runtime.release_calls == 3  # reset, sync-off finally, error fallback


def test_base_done_immediately_releases_leader_feedback(monkeypatch):
    runtime = _Runtime()
    base_env = DualYamJointEnv(
        override_cfg={
            "is_dummy": True,
            "dummy_camera_names": ["top_rgb"],
            "max_num_steps": 1,
        },
        runtime=runtime,
    )
    monkeypatch.setattr(base_env, "_pace", lambda: None)
    env = DualYamLeaderIntervention(
        base_env,
        {
            "wait_for_record_button": False,
            "button_debounce_s": 0.0,
            "unsynced_action_source": "policy",
        },
    )
    env.reset()
    runtime.buttons = (True, False)

    _, _, terminated, truncated, info = env.step(np.zeros(14))

    assert not terminated
    assert truncated
    assert not env._sync_enabled
    assert not info["yam_sync_enabled"]
    assert runtime.release_calls == 2  # reset plus done handoff


def test_manual_record_boundary_preserves_sync_across_reset(monkeypatch, capsys):
    runtime = _Runtime()
    base_env = DualYamJointEnv(
        override_cfg={"is_dummy": True, "dummy_camera_names": ["top_rgb"]},
        runtime=runtime,
    )
    monkeypatch.setattr(base_env, "_pace", lambda: None)
    env = DualYamLeaderIntervention(
        base_env,
        {
            "wait_for_record_button": False,
            "button_debounce_s": 0.0,
            "preserve_sync_between_episodes": True,
            "unsynced_action_source": "hold",
        },
    )
    env.reset()

    runtime.buttons = (True, False)
    env.step(np.zeros(14))
    runtime.buttons = (False, False)
    env.step(np.zeros(14))
    runtime.buttons = (False, True)

    _, _, terminated, truncated, info = env.step(np.zeros(14))

    assert terminated
    assert not truncated
    output = capsys.readouterr().err
    assert "[录制中]" in output
    assert "[录制结束]" in output
    assert "[后台保存完成]" not in output
    assert info["manual_done"]
    assert info["yam_sync_enabled"]
    assert env._sync_enabled
    assert runtime.release_calls == 1

    env.reset()

    assert env._sync_enabled
    assert runtime.engage_calls == 1
    assert runtime.release_calls == 1


def test_episode_qpos_reset_takes_ownership_from_preserved_leader_sync(monkeypatch):
    runtime = _Runtime()
    target = [0.1] * 6 + [1.0] + [0.2] * 6 + [1.0]
    base_env = DualYamJointEnv(
        override_cfg={
            "is_dummy": True,
            "dummy_camera_names": ["top_rgb"],
            "reset": {
                "enabled": True,
                "mode": "episode",
                "left_qpos": target[:7],
                "right_qpos": target[7:],
                "duration_s": 0.0,
                "timeout_s": 1.0,
            },
        },
        runtime=runtime,
    )
    monkeypatch.setattr(base_env, "_pace", lambda: None)
    env = DualYamLeaderIntervention(
        base_env,
        {
            "wait_for_record_button": False,
            "button_debounce_s": 0.0,
            "preserve_sync_between_episodes": True,
            "unsynced_action_source": "hold",
        },
    )
    env.reset()
    runtime.buttons = (True, False)
    env.step(np.zeros(14))
    runtime.buttons = (False, False)
    env.step(np.zeros(14))
    runtime.buttons = (False, True)
    env.step(np.zeros(14))

    env.reset()

    assert len(runtime.moves) == 2
    assert not env._sync_enabled
    assert runtime.release_calls == 3


def test_early_reset_still_releases_sync_in_continuous_mode(monkeypatch):
    runtime = _Runtime()
    base_env = DualYamJointEnv(
        override_cfg={"is_dummy": True, "dummy_camera_names": ["top_rgb"]},
        runtime=runtime,
    )
    monkeypatch.setattr(base_env, "_pace", lambda: None)
    env = DualYamLeaderIntervention(
        base_env,
        {
            "wait_for_record_button": False,
            "button_debounce_s": 0.0,
            "preserve_sync_between_episodes": True,
            "unsynced_action_source": "hold",
        },
    )
    env.reset()
    runtime.buttons = (True, False)
    env.step(np.zeros(14))

    env.reset()

    assert not env._sync_enabled
    # One hold initializes the base environment; the second safely disables
    # synchronization when reset occurs before an episode boundary.
    assert runtime.hold_calls == 2
    assert runtime.release_calls == 3


@pytest.mark.parametrize("choice", ["keep", "discard"])
def test_foot_review_keeps_teleop_live_without_recording(monkeypatch, choice):
    from rlinf.envs.real.utils import foot_switch

    class Pedal:
        choice = None
        begins = 0
        closed = False

        def __init__(self, *args, **kwargs):
            pass

        def begin_review(self):
            self.begins += 1
            self.choice = None

        def poll(self):
            result, self.choice = self.choice, None
            return result

        def close(self):
            self.closed = True

    pedal = Pedal()
    from rlinf.envs.real.yam import leader_intervention

    monkeypatch.setattr(foot_switch, "FootSwitch", lambda *a, **kw: pedal)
    monkeypatch.setattr(leader_intervention, "FootSwitch", lambda *a, **kw: pedal)
    runtime = _Runtime()
    base = DualYamJointEnv(
        override_cfg={"is_dummy": True, "dummy_camera_names": ["top_rgb"]},
        runtime=runtime,
    )
    monkeypatch.setattr(base, "_pace", lambda: None)
    env = DualYamLeaderIntervention(
        base,
        {
            "wait_for_record_button": False,
            "sync_on_reset": True,
            "preserve_sync_between_episodes": True,
            "button_debounce_s": 0.0,
            "foot_switch_device": "/dev/test-foot-switch",
        },
    )
    try:
        env.reset()
        runtime.buttons = (False, True)
        _, _, term, trunc, info = env.step(np.zeros(14))
        assert not term and not trunc
        assert info["episode_review_pending"] and not info["pre_record"]
        assert pedal.begins == 1
        for buttons in [(False, False), (False, True), (False, False)]:
            runtime.buttons = buttons
            runtime.leader_action += 0.01
            _, _, term, trunc, info = env.step(np.zeros(14))
            assert not term and not trunc
            assert info["episode_review_pending"] and info["pre_record"]
            assert info["keyboard_event"] != "start"
            np.testing.assert_allclose(runtime.commands[-1], runtime.leader_action)
        pedal.choice = choice
        _, reward, term, trunc, info = env.step(np.zeros(14))
        assert term and not trunc and info["pre_record"]
        assert not info["episode_review_pending"]
        assert info["episode_discarded"] == (choice == "discard")
        assert info["manual_done"] == (choice == "keep")
        assert reward == float(choice == "keep")
        assert info["yam_sync_enabled"]
    finally:
        env.close()
    assert pedal.closed


def _return_env(monkeypatch, *, recording=True):
    from rlinf.envs.real.yam import leader_intervention

    class Pedal:
        choice = None
        closed = False

        def poll(self):
            choice, self.choice = self.choice, None
            return choice

        def begin_review(self):
            self.choice = None

        def close(self):
            self.closed = True

    pedal = Pedal()
    monkeypatch.setattr(leader_intervention, "FootSwitch", lambda *a, **kw: pedal)
    runtime = _Runtime()
    runtime.leader_action[:] = 0.6
    runtime.leader_action[[6, 13]] = [0.4, 0.7]
    base = DualYamJointEnv(
        override_cfg={
            "is_dummy": True,
            "dummy_camera_names": ["top_rgb"],
            "reset": {
                "enabled": True,
                "mode": "manual",
                "left_qpos": [0.1] * 6 + [1.0],
                "right_qpos": [0.2] * 6 + [1.0],
                "duration_s": 0.1,
                "max_joint_delta": 0.04,
                "timeout_s": 5.0,
            },
        },
        runtime=runtime,
    )
    monkeypatch.setattr(base, "_pace", lambda: None)
    env = DualYamLeaderIntervention(
        base,
        {
            "wait_for_record_button": not recording,
            "sync_on_reset": True,
            "preserve_sync_between_episodes": True,
            "button_debounce_s": 0.0,
            "foot_switch_device": "/dev/test-foot-switch",
            "foot_switch_reset_key": 99,
        },
    )
    env.reset(options={"skip_wait_for_start": True})
    return env, runtime, pedal


@pytest.mark.parametrize("recording", [True, False])
def test_pedal_return_is_per_step_preserves_episode_and_gripper(monkeypatch, recording):
    env, runtime, pedal = _return_env(monkeypatch, recording=recording)
    try:
        pedal.choice = "reset"
        infos = []
        start = runtime.state.copy()
        for tick in range(50):
            if tick == 1:
                runtime.buttons = (False, True)  # Must not cut the return short.
                pedal.choice = "reset"  # Must not restart the trajectory.
            else:
                runtime.buttons = (False, False)
            _, _, terminated, truncated, info = env.step(np.zeros(14))
            infos.append(info)
            assert not terminated and not truncated
            assert info["pre_record"] == (not recording)
            assert not info["record_reset"] and not info["episode_review_pending"]
            np.testing.assert_allclose(info["intervene_action"], runtime.commands[-1])
            if env._return_start is None:
                break
        assert 2 < len(infos) < 50
        assert runtime.moves == []  # No blocking move_to calls.
        joints = [*range(6), *range(7, 13)]
        trajectory = np.stack([start, *runtime.commands])
        assert np.abs(np.diff(trajectory[:, joints], axis=0)).max() <= 0.04 + 1e-9
        np.testing.assert_allclose(
            runtime.state[joints], env.env.config.reset.as_vector()[joints]
        )
        np.testing.assert_allclose(runtime.leader_action[joints], runtime.state[joints])
        np.testing.assert_allclose(
            trajectory[:, [6, 13]], np.tile([0.4, 0.7], (len(trajectory), 1))
        )
        assert env._sync_enabled
        runtime.buttons = (False, True)
        _, _, _, _, info = env.step(np.zeros(14))
        assert info["keyboard_event"] == ("end_pending" if recording else "start")
        if recording:
            pedal.choice = "reset"
            count = len(runtime.leader_commands)
            env.step(np.zeros(14))
            assert len(runtime.leader_commands) == count  # Review ignores middle.
    finally:
        env.close()


def test_yellow_button_cancels_return_without_ending_recording(monkeypatch):
    env, runtime, pedal = _return_env(monkeypatch)
    try:
        pedal.choice = "reset"
        env.step(np.zeros(14))
        count = len(runtime.leader_commands)
        runtime.buttons = (True, False)
        _, _, term, trunc, info = env.step(np.zeros(14))
        assert env._return_start is None and not env._sync_enabled
        assert len(runtime.leader_commands) == count
        assert not term and not trunc and not info["pre_record"]
        assert not info["record_reset"]
    finally:
        env.close()


def test_return_timeout_holds_and_keeps_recording(monkeypatch):
    env, runtime, pedal = _return_env(monkeypatch)
    try:
        pedal.choice = "reset"
        env.step(np.zeros(14))
        env._return_started_s -= 6.0
        count = len(runtime.leader_commands)
        _, _, term, trunc, info = env.step(np.zeros(14))
        assert env._return_start is None and not env._sync_enabled
        assert len(runtime.leader_commands) == count
        assert not term and not trunc and not info["pre_record"]
    finally:
        env.close()


def test_unconfirmed_pose_and_sync_off_do_not_command_leaders(monkeypatch):
    env, runtime, pedal = _return_env(monkeypatch)
    try:
        env.env.config.reset.enabled = False
        pedal.choice = "reset"
        env.step(np.zeros(14))
        assert not runtime.leader_commands
        env.env.config.reset.enabled = True
        env._disable_sync()
        pedal.choice = "reset"
        env.step(np.zeros(14))
        assert not runtime.leader_commands
    finally:
        env.close()


def test_startup_wait_loop_finishes_return_before_accepting_white(monkeypatch):
    env, runtime, pedal = _return_env(monkeypatch, recording=False)
    original_read = runtime.read_leader_action
    reads = 0

    def read():
        nonlocal reads
        reads += 1
        runtime.buttons = (False, False)
        if reads == 1:
            pedal.choice = "reset"
        elif reads == 2 or (reads > 2 and env._return_start is None):
            runtime.buttons = (False, True)
        if reads > 60:
            raise AssertionError("startup wait failed to accept the white button")
        return original_read()

    monkeypatch.setattr(runtime, "read_leader_action", read)
    monkeypatch.setattr(
        "rlinf.envs.real.yam.leader_intervention.time.sleep", lambda _: None
    )
    try:
        env._wait_for_record_start()
        assert 3 < reads < 60
        assert runtime.leader_commands and not runtime.moves
        assert env._return_start is None
        assert not env._recording  # Caller sets recording only after this returns.
        assert env.env._num_steps == 0  # Waiting doesn't consume episode steps.
    finally:
        env.close()


def test_close_releases_pedal_even_when_return_hold_fails(monkeypatch):
    env, runtime, pedal = _return_env(monkeypatch)
    pedal.choice = "reset"
    env.step(np.zeros(14))
    runtime.fail_hold = True
    with pytest.raises(RuntimeError, match="hold failed"):
        env.close()
    assert pedal.closed
    assert runtime.close_calls == 1
    assert env._return_start is None
