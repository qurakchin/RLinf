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

"""Two independent PICO controllers driving YAM joints through checked IK."""

from __future__ import annotations

import os
import select
import time
from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from rlinf.utils.logging import get_logger

from .config import YamPicoConfig
from .kinematics import YamIKResult, YamKinematicsAdapter


def _delta_to_tcp_pose(
    action: np.ndarray,
    tcp_pose: np.ndarray,
    action_scale: np.ndarray,
    *,
    clip_motion: bool = True,
) -> np.ndarray:
    """Compose a normalized motion delta with the measured TCP pose.

    Returns xyz in meters followed by an xyzw quaternion. ``clip_motion``
    caps the spatial rotation error; the VR path passes ``False`` so the
    operator's full wrist angle reaches IK, where the joint-step limits
    decide how much of it one control tick may execute.
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    action_scale = np.asarray(action_scale, dtype=np.float64)
    target_pos = np.asarray(tcp_pose[:3], dtype=np.float64) + action[:3] * float(
        action_scale[0]
    )
    rot_action = np.asarray(action[3:6], dtype=np.float64)
    norm = float(np.linalg.norm(rot_action))
    if clip_motion and norm > 1.0:
        rot_action = rot_action / norm
    target_rot = Rotation.from_rotvec(rot_action * float(action_scale[1])) * (
        Rotation.from_quat(np.asarray(tcp_pose[3:7], dtype=np.float64))
    )
    return np.concatenate([target_pos, target_rot.as_quat()])


class _YamPicoArm:
    """One controller composed onto one YAM arm's measured TCP pose.

    The transport in :mod:`rlinf.robotics.parts.transports.pico` owns the
    socket, the base calibration, and the motion accumulated since the grip
    was engaged. This class adds what the YAM VR path needs and the shared
    Cartesian devices do not: a *cumulative* TCP target, anchored where the
    arm stood when the grip closed, whose rotation reaches equal to the
    operator's rotation rather than a per-step slice of it.
    """

    def __init__(self, **expert_kwargs: Any) -> None:
        from rlinf.robotics.parts.transports.pico import PicoExpert

        self.hand = str(expert_kwargs["hand"])
        self._expert = PicoExpert(**expert_kwargs)
        self._ref_tcp_pos: np.ndarray | None = None
        self._ref_tcp_rot: Rotation | None = None

    @property
    def ready(self) -> bool:
        """Whether the controller stream is fresh enough to command from."""
        return self._expert.ready

    def read_buttons(self) -> dict[str, bool]:
        """Return raw controller buttons, for record and discard edges."""
        return self._expert.get_buttons()

    def reset_reference(self) -> None:
        """Forget the TCP pose the last grip was anchored to."""
        self._ref_tcp_pos = None
        self._ref_tcp_rot = None

    def stop(self) -> None:
        self._expert.stop()

    def get_action(
        self,
        tcp_pose: np.ndarray,
        action_scale: np.ndarray,
        *,
        gripper_enabled: bool = True,
        clip_motion: bool = True,
    ) -> tuple[np.ndarray, bool, dict[str, Any]]:
        """Encode the controller target as a scaled spatial pose error.

        With ``clip_motion=False``, motion components may exceed ``[-1, 1]``;
        compose them with :func:`_delta_to_tcp_pose` using the same flag to
        recover the full target and let the robot controller limit motion.
        """
        reading = self._expert.get_reading()
        info = self._describe(reading)
        if not reading.get("held", False):
            # The transport reports a missing calibration, a stale stream, and
            # an invalid pose by releasing the grip, so this covers them all.
            self.reset_reference()
            return np.zeros(7, dtype=np.float32), False, info

        pose = np.asarray(tcp_pose, dtype=np.float64).reshape(-1)
        if self._ref_tcp_pos is None:
            self._ref_tcp_pos = pose[:3].copy()
            self._ref_tcp_rot = Rotation.from_quat(pose[3:7])

        target_pos = self._ref_tcp_pos + np.asarray(
            reading["position_delta"], dtype=np.float64
        )
        target_rot = (
            Rotation.from_rotvec(
                np.asarray(reading["rotation_delta"], dtype=np.float64)
            )
            * self._ref_tcp_rot
        )

        scale = np.asarray(action_scale, dtype=np.float64)
        current_rot = Rotation.from_quat(pose[3:7])
        delta_pos = (target_pos - pose[:3]) / float(scale[0])
        delta_rotvec = (target_rot * current_rot.inv()).as_rotvec()
        max_rot = float(scale[1])
        if max_rot > 1e-9:
            angle = float(np.linalg.norm(delta_rotvec))
            if clip_motion and angle > max_rot:
                delta_rotvec = delta_rotvec * (max_rot / angle)
            delta_rot = delta_rotvec / max_rot
        else:
            delta_rot = np.zeros(3, dtype=np.float64)

        action = np.concatenate((delta_pos, delta_rot))
        if clip_motion:
            action = np.clip(action, -1.0, 1.0)
        if gripper_enabled:
            grip = 0.0
            if reading.get("grip_close", False):
                grip = -1.0
            elif reading.get("grip_open", False):
                grip = 1.0
            action = np.concatenate((action, np.array([grip], dtype=np.float64)))
        return action.astype(np.float32), True, info

    def _describe(self, reading: dict[str, Any]) -> dict[str, Any]:
        """Report controller state under the collector's field names."""
        info: dict[str, Any] = {
            "pico_active": bool(reading.get("held", False)),
            "pico_hand": self.hand,
        }
        if reading.get("stale"):
            return {**info, "pico_ready": False, "pico_stale": True}

        info["pico_ready"] = bool(reading.get("ready", False))
        info["pico_calibrated"] = bool(reading.get("calibrated", False))
        info["pico_control_value"] = reading.get("control_value", 0.0)
        if reading.get("invalid_pose"):
            info["pico_invalid_pose"] = True
        if reading.get("held", False):
            close = bool(reading.get("grip_close", False))
            opened = bool(reading.get("grip_open", False))
            info["pico_gripper_close_pressed"] = close
            info["pico_gripper_open_pressed"] = opened
            info["pico_gripper_action"] = -1.0 if close else (1.0 if opened else 0.0)
            info["pico_gripper_close"] = close
        return info


class _RecordingKeyboard:
    """Poll an explicitly selected Linux keyboard without another daemon."""

    def __init__(self) -> None:
        from evdev import InputDevice, ecodes

        path = os.environ.get("RLINF_KEYBOARD_DEVICE")
        if not path:
            raise ValueError(
                "pico.keyboard_enabled requires RLINF_KEYBOARD_DEVICE=/dev/input/by-id/..."
            )
        self.device = InputDevice(path)
        self.keys = {ecodes.KEY_R: "record", ecodes.KEY_X: "discard"}
        self.key_type = ecodes.EV_KEY

    def poll(self) -> set[str]:
        """Read R/X press edges, ignoring key repeats."""
        if not select.select([self.device], [], [], 0)[0]:
            return set()
        return {
            self.keys[e.code]
            for e in self.device.read()
            if e.type == self.key_type and e.value == 1 and e.code in self.keys
        }

    def close(self) -> None:
        """Release the input device."""
        self.device.close()


class DualYamPicoIntervention(gym.Wrapper):
    """Collection-only VR wrapper: inactive arms hold, never follow policy zeros.

    Initialization is lazy. Models/experts are prepared before base reset opens
    cameras and followers. Explicitly injected experts/kinematics support tests.
    """

    def __init__(
        self,
        env: gym.Env,
        config: dict[str, Any] | YamPicoConfig | None = None,
        *,
        experts: dict[str, Any] | None = None,
        kinematics: dict[str, Any] | None = None,
        keyboard: Any | None = None,
    ) -> None:
        super().__init__(env)
        self.config = (
            config
            if isinstance(config, YamPicoConfig)
            else YamPicoConfig(**dict(config or {}))
        )
        if not env.unwrapped.config.enforce_runtime_joint_limits:
            raise ValueError("YAM VR requires enforce_runtime_joint_limits=true")
        self.experts = dict(experts or {})
        self.kinematics = dict(kinematics or {})
        for collection in (self.experts, self.kinematics):
            if collection and set(collection) != {"left", "right"}:
                raise ValueError("YAM PICO requires both left and right instances")
        self._keyboard = keyboard
        self._recording = False
        self._preserve_reference = False
        self._fault = "release_grips"
        self._last_ik_warning = -float("inf")
        self._last_step_log = dict.fromkeys(("left", "right"), -float("inf"))
        self._timing_started: float | None = None
        self._timing_peaks: dict[str, float] = {}
        self._timing_count = 0
        self._timing_faults = 0
        self._last_tick_return_s: float | None = None
        self._grippers: np.ndarray | None = None
        self._idle_joint_targets: dict[str, np.ndarray] = {}
        self._previous_buttons: tuple[bool, bool] | None = None
        self._last_edges = [-float("inf"), -float("inf")]
        self._closed = False
        self._initialized = False
        self._logger = get_logger()

    def _initialize(self) -> None:
        if self._closed:
            raise RuntimeError("YAM PICO wrapper is closed")
        if self._initialized:
            return
        if not self.kinematics:
            base = self.env.unwrapped
            for index, side in enumerate(("left", "right")):
                device = getattr(base.hardware_config, f"{side}_follower")
                self.kinematics[side] = YamKinematicsAdapter(
                    arm_type=getattr(device, "arm_type", "yam"),
                    gripper_type=getattr(device, "gripper_type", "flexible_4310"),
                    config=self.config.ik,
                    joint_lower=base.config.joint_limit_min[index],
                    joint_upper=base.config.joint_limit_max[index],
                )
        if not self.experts:
            for side in ("left", "right"):
                self.experts[side] = _YamPicoArm(**self.config.expert_kwargs(side))
        if self.config.keyboard_enabled and self._keyboard is None:
            self._keyboard = _RecordingKeyboard()
        self._initialized = True

    def _latch(self, reason: str) -> None:
        if self._fault != reason:
            self._logger.warning("YAM VR holding both arms: %s", reason)
        self._fault = reason
        self._preserve_reference = False
        self._grippers = None
        self._idle_joint_targets.clear()
        for expert in self.experts.values():
            expert.reset_reference()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Keep teleoperation ticking while waiting for a fresh recording start."""
        try:
            self._initialize()
            preserve = (
                self._preserve_reference
                and not self.env.unwrapped.will_reset_to_configured_qpos(options)
            )
            if not preserve:
                self._latch("release_grips")
                self._previous_buttons = None
            self._preserve_reference = False
            self._recording = False
            obs, info = self.env.reset(seed=seed, options=options)
            if self.config.wait_for_record_button and not (options or {}).get(
                "skip_wait_for_start", False
            ):
                while not self._recording:
                    obs, _, _, _, info = self._tick(preview=True)
            return obs, self._decorate(info, None, False)
        except BaseException:
            self.close()
            raise

    def _compute_action(self) -> tuple[np.ndarray, dict[str, Any]]:
        started = time.monotonic()
        measured = self.env.unwrapped.get_hold_action().astype(np.float64)
        if self._grippers is None:
            self._grippers = measured[[6, 13]].copy()
        target = measured.copy()
        scale = np.array(
            [self.config.max_position_delta, self.config.max_rotation_delta, 1.0]
        )
        infos: dict[str, Any] = {}
        results = {}
        poses = {}
        reason = None
        ik_fault = None
        for i, side in enumerate(("left", "right")):
            arm = measured[i * 7 : i * 7 + 7]
            infos[f"{side}_joint_step_fraction"] = 1.0
            infos[f"{side}_limiting_joint"] = 0
            infos[f"{side}_ik_max_joint_delta"] = 0.0
            infos[f"{side}_ik_attempts"] = 0
            infos[f"{side}_ik_target_fraction"] = 0.0
            infos[f"{side}_ik_initial_fault"] = None
            pose = self.kinematics[side].fk(arm[:6], arm[6])
            poses[side] = pose
            tcp = np.concatenate(
                [pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()]
            )
            action, replaced, info = self.experts[side].get_action(
                tcp, scale, gripper_enabled=True, clip_motion=False
            )
            results[side] = (action, replaced, info)
            infos.update({f"{side}_{key}": value for key, value in info.items()})
            if (
                not info.get("pico_ready", False)
                or info.get("pico_invalid_pose", False)
                or not info.get("pico_calibrated", False)
            ):
                reason = f"{side}:unavailable"
        if reason:
            self._latch(reason)
        elif self._fault is not None:
            # Never re-engage a still-held grip after data loss or reset.
            released = all(
                info.get("pico_control_value", float("inf"))
                < self.config.control_threshold
                for _, _, info in results.values()
            )
            for expert in self.experts.values():
                expert.reset_reference()
            if released:
                self._fault = None
                self._idle_joint_targets = {
                    side: measured[i * 7 : i * 7 + 6].copy()
                    for i, side in enumerate(("left", "right"))
                }
            infos["yam_pico_fault"] = self._fault
            return measured, infos

        if self._fault is None:
            for i, side in enumerate(("left", "right")):
                action, replaced, _ = results[side]
                if not replaced:
                    # Latch the release pose once so feedback drift does not
                    # continuously move the position controller's target.
                    if side not in self._idle_joint_targets:
                        self._idle_joint_targets[side] = measured[
                            i * 7 : i * 7 + 6
                        ].copy()
                    target[i * 7 : i * 7 + 6] = self._idle_joint_targets[side]
                    self._grippers[i] = measured[i * 7 + 6]
                    continue
                self._idle_joint_targets.pop(side, None)
                action = np.asarray(action, dtype=np.float64)
                if action.shape != (7,) or not np.all(np.isfinite(action)):
                    self._latch(f"{side}:invalid_action")
                    break
                current = poses[side]
                tcp = np.r_[
                    current[:3, 3], Rotation.from_matrix(current[:3, :3]).as_quat()
                ]
                target_tcp = _delta_to_tcp_pose(action, tcp, scale, clip_motion=False)
                pose = np.eye(4)
                pose[:3, 3] = target_tcp[:3]
                pose[:3, :3] = Rotation.from_quat(target_tcp[3:]).as_matrix()
                arm = measured[i * 7 : i * 7 + 7]
                result, attempts, fraction, first_reason = (
                    self._solve_with_backtracking(side, pose, current, arm)
                )
                infos[f"{side}_ik_attempts"] = attempts
                infos[f"{side}_ik_target_fraction"] = fraction
                infos[f"{side}_ik_initial_fault"] = first_reason
                infos[f"{side}_ik_elapsed_s"] = result.elapsed_s
                infos[f"{side}_ik_position_error"] = result.position_error
                infos[f"{side}_ik_rotation_error"] = result.rotation_error
                infos[f"{side}_ik_fault"] = (
                    result.reason if not result.success else None
                )
                if not result.success:
                    # Hold only this arm, retaining references for the next
                    # frame. The other arm can still execute a valid solution.
                    ik_fault = f"{side}:ik:{result.reason}"
                    self._grippers[i] = measured[i * 7 + 6]
                    now = time.monotonic()
                    if now - self._last_ik_warning >= 1.0:
                        self._logger.warning(
                            "YAM VR holding %s arm this frame, retrying: %s",
                            side,
                            ik_fault,
                        )
                        self._last_ik_warning = now
                    continue
                # Interpolate toward the full or successfully reduced IK target.
                # A tiny TCP target ahead of lagging feedback can stall motion.
                difference = result.q_target - arm[:6]
                limits = self.env.unwrapped.config.joint_step_limits
                demand = np.abs(difference) / limits
                limiting_joint = int(np.argmax(demand))
                fraction = min(1.0, 1.0 / max(float(demand[limiting_joint]), 1e-12))
                target[i * 7 : i * 7 + 6] = arm[:6] + fraction * difference
                infos[f"{side}_joint_step_fraction"] = fraction
                infos[f"{side}_limiting_joint"] = (
                    limiting_joint + 1 if fraction < 1.0 else 0
                )
                infos[f"{side}_ik_max_joint_delta"] = float(np.max(np.abs(difference)))
                now = time.monotonic()
                if fraction < 1.0 and now - self._last_step_log[side] >= 1.0:
                    self._logger.info(
                        "YAM VR %s joint step limited: fraction=%.3f, joint=J%d, "
                        "required=%.3f rad, limit=%.3f rad",
                        side,
                        fraction,
                        limiting_joint + 1,
                        abs(difference[limiting_joint]),
                        limits[limiting_joint],
                    )
                    self._last_step_log[side] = now
                if action[6] < 0:
                    self._grippers[i] = 0.0
                elif action[6] > 0:
                    self._grippers[i] = 1.0
                target[i * 7 + 6] = self._grippers[i]
        # Reject solutions computed from a now-stale stream or a slow solve.
        if self._fault is None and (
            time.monotonic() - started > self.config.max_tick_s
            or not all(e.ready for e in self.experts.values())
        ):
            self._latch("stale_after_ik")
        infos["yam_pico_fault"] = self._fault or ik_fault
        return (measured if self._fault else target), infos

    def _solve_with_backtracking(
        self, side: str, target: np.ndarray, current: np.ndarray, arm: np.ndarray
    ) -> tuple[YamIKResult, int, float, str | None]:
        """Retry geometric IK failures toward the same measured TCP reference.

        Attempts share max_solve_s; timing is checked between synchronous SDK
        calls and on return, rather than preempting a call. Each retry halves
        both translation and the shortest spatial rotation from current.
        """
        started = time.monotonic()
        result = self.kinematics[side].solve(target, arm[:6].copy(), arm[6])
        first_reason = result.reason
        attempts, fraction = 1, 1.0
        delta_rotation = None
        while True:
            elapsed = time.monotonic() - started
            if elapsed > self.config.ik.max_solve_s:
                result = replace(result, success=False, reason="solve_timeout")
                break
            if (
                result.success
                or result.reason not in {"not_converged", "residual", "joint_limits"}
                or attempts > self.config.ik_backtrack_attempts
            ):
                break
            if delta_rotation is None:
                delta_rotation = Rotation.from_matrix(
                    target[:3, :3] @ current[:3, :3].T
                ).as_rotvec()
            fraction *= 0.5
            intermediate = current.copy()
            intermediate[:3, 3] += fraction * (target[:3, 3] - current[:3, 3])
            intermediate[:3, :3] = (
                Rotation.from_rotvec(fraction * delta_rotation).as_matrix()
                @ current[:3, :3]
            )
            result = self.kinematics[side].solve(intermediate, arm[:6].copy(), arm[6])
            attempts += 1
        return replace(result, elapsed_s=elapsed), attempts, fraction, first_reason

    def _events(self) -> tuple[bool, bool]:
        left = self.experts["left"].read_buttons()
        right = self.experts["right"].read_buttons()
        buttons = tuple(
            bool(left.get(key, False) or right.get(key, False))
            for key in (self.config.record_button, self.config.discard_button)
        )
        keyboard = self._keyboard.poll() if self._keyboard else set()
        now = time.monotonic()
        previous = self._previous_buttons or buttons
        edges = []
        for i, key in enumerate(("record", "discard")):
            edge = (
                (buttons[i] and not previous[i]) or key in keyboard
            ) and now - self._last_edges[i] >= self.config.button_debounce_s
            if edge:
                self._last_edges[i] = now
            edges.append(edge)
        self._previous_buttons = buttons
        return tuple(edges)

    def _decorate(
        self, info: dict[str, Any], event: str | None, record_reset: bool
    ) -> dict[str, Any]:
        phase = "rec" if self._recording else "pre"
        info.update(
            {
                "pre_record": not self._recording,
                "record_reset": record_reset,
                "keyboard_phase": phase,
                "keyboard_event": event,
                "episode_control_phase": phase,
                "episode_control_event": event,
                "segment_advance": False,
            }
        )
        return info

    def _report_timing(
        self, info: dict[str, Any], started: float, computed: float
    ) -> None:
        """Report one-second peak timings; these do not change control targets."""
        finished = time.monotonic()
        info["yam_pico_compute_s"] = computed - started
        info["yam_between_ticks_s"] = (
            started - self._last_tick_return_s
            if self._last_tick_return_s is not None
            else 0.0
        )
        info["yam_tick_s"] = finished - started
        if self._timing_started is None:
            self._timing_started = started
        self._timing_count += 1
        self._timing_faults += int(info.get("yam_pico_fault") is not None)
        metrics = {
            "gap": info.get("yam_command_interval_s", 0.0),
            "compute": info["yam_pico_compute_s"],
            "ik": sum(
                info.get(f"{side}_ik_elapsed_s", 0.0) for side in ("left", "right")
            ),
            "pace": info.get("yam_pace_s", 0.0),
            "send": info.get("yam_command_s", 0.0),
            "state": info.get("yam_state_read_s", 0.0),
            "cameras": info.get("yam_camera_read_s", 0.0),
            "outside": info["yam_between_ticks_s"],
        }
        for name, value in metrics.items():
            self._timing_peaks[name] = max(self._timing_peaks.get(name, 0.0), value)
        elapsed = finished - self._timing_started
        if elapsed >= 1.0:
            self._logger.info(
                "YAM VR timing: ticks=%d, rate=%.1f Hz, fault_ticks=%d; peak_ms %s",
                self._timing_count,
                self._timing_count / elapsed,
                self._timing_faults,
                " ".join(
                    f"{key}={value * 1000:.1f}"
                    for key, value in self._timing_peaks.items()
                ),
            )
            self._timing_started = finished
            self._timing_count = self._timing_faults = 0
            self._timing_peaks.clear()
        self._last_tick_return_s = finished

    def _tick(self, *, preview: bool) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        try:
            started = time.monotonic()
            target, pico_info = self._compute_action()
            computed = time.monotonic()
            record, discard = self._events()
            if preview:
                obs, info = self.env.unwrapped.teleop_tick(target)
                reward, terminated, truncated = 0.0, False, False
            else:
                obs, reward, terminated, truncated, info = self.env.step(target)
            if info.get("action_rejected") is not None:
                self._latch(f"runtime:{info['action_rejected']}")
            frame_fault = self._fault or pico_info.get("yam_pico_fault")
            event = None
            reset_record = False
            manual_done = False
            if frame_fault is not None or discard or truncated:
                reset_record = self._recording
                self._recording = False
                event = "abort" if reset_record else None
                reward = 0.0
            elif record:
                if self._recording:
                    event = "end_success"
                    reward, terminated, manual_done = 1.0, True, True
                    self._preserve_reference = True
                else:
                    self._recording = True
                    reset_record = True
                    event = "start"
            info.update(pico_info)
            info["yam_pico_fault"] = frame_fault
            info["intervene_action"] = np.asarray(
                info["accepted_action"], dtype=np.float32
            ).copy()
            info["intervene_flag"] = np.ones(1, dtype=bool)
            info["intervened"] = True
            info["manual_done"] = manual_done
            info["success"] = manual_done
            for side in ("left", "right"):
                info[side] = (
                    self._fault is None
                    and pico_info.get(f"{side}_ik_fault") is None
                    and bool(pico_info.get(f"{side}_pico_active", False))
                )
            info["pico_active"] = info["left"] or info["right"]
            if (terminated or truncated) and not manual_done:
                self._latch("episode_ended")
                self.env.unwrapped.runtime.emergency_hold()
            if self.config.log_control_timing:
                self._report_timing(info, started, computed)
            return (
                obs,
                reward,
                terminated,
                truncated,
                self._decorate(info, event, reset_record),
            )
        except BaseException:
            self._latch("control_error")
            try:
                self.env.unwrapped.runtime.emergency_hold()
            except Exception:
                self._logger.exception("YAM VR emergency hold failed")
            try:
                self.close()
            except Exception:
                self._logger.exception("YAM VR cleanup failed after control error")
            raise

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute a VR/hold step and expose accepted joint targets to collection."""
        del action
        if not self._initialized or self._closed:
            raise RuntimeError("reset YAM PICO before stepping")
        return self._tick(preview=False)

    def close(self) -> None:
        """Hold/close the base env and stop both independent subscriptions."""
        if self._closed:
            return
        errors = []
        for resource in [self.env, *self.experts.values(), self._keyboard]:
            if resource is None:
                continue
            try:
                if resource in self.experts.values():
                    resource.stop()
                else:
                    resource.close()
            except Exception as exc:
                errors.append(exc)
        self._closed = not errors
        if errors:
            raise RuntimeError("YAM PICO cleanup failed") from errors[0]
