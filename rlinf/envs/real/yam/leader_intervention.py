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

"""Teaching-handle teleoperation and episode control for dual YAM."""

from __future__ import annotations

import math
import sys
import time
from typing import Any

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from rlinf.envs.real.wrappers.episode.foot_switch import FootSwitch

from .config import YamLeaderInterventionConfig
from .dual_yam_joint_env import DualYamJointEnv


class DualYamLeaderIntervention(gym.Wrapper):
    """Drive followers from leaders and map handle buttons to collection state.

    Either leader's top button toggles follower synchronization. Either second
    button starts or successfully ends an episode. Button handling is rising-edge
    based, so holding a button cannot repeatedly toggle state.
    """

    def __init__(
        self,
        env: DualYamJointEnv,
        config: "YamLeaderInterventionConfig | dict[str, Any] | None" = None,
    ) -> None:
        super().__init__(env)
        self.config = (
            config
            if isinstance(config, YamLeaderInterventionConfig)
            else YamLeaderInterventionConfig(**dict(config or {}))
        )
        self._recording = False
        self._sync_enabled = False
        self._review_pending = False
        self._preserve_sync_on_next_reset = False
        self._previous_buttons = (False, False)
        self._last_button_edge_s = [-math.inf, -math.inf]
        self._foot_switch: FootSwitch | None = None
        self._return_start: np.ndarray | None = None
        self._return_step = 0
        self._return_steps = 0
        self._return_started_s = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset episode state, then optionally wait for the record button."""
        try:
            self._ensure_foot_switch()
            if self._return_start is not None:
                self._disable_sync()
            self._review_pending = False
            reset_will_move = self._base_env.will_reset_to_configured_qpos(options)
            preserve_sync = bool(
                self._sync_enabled
                and self._preserve_sync_on_next_reset
                and not reset_will_move
            )
            self._preserve_sync_on_next_reset = False
            # A caller may reset early, before a normal done transition. Never
            # carry bilateral feedback across it unless the preceding manual
            # record boundary explicitly requested legacy continuous teleop.
            # A configured follower reset also always takes ownership first.
            if self._sync_enabled and not preserve_sync:
                self._disable_sync()
            observation, info = self.env.reset(seed=seed, options=options)
            runtime = self._base_env.runtime
            runtime.connect_leaders()
            if not preserve_sync:
                runtime.release_leader_feedback()
            _, self._previous_buttons = runtime.read_leader_action()
            self._last_button_edge_s = [-math.inf, -math.inf]
            self._recording = not self.config.wait_for_record_button
            self._sync_enabled = preserve_sync

            if self.config.sync_on_reset and not self._sync_enabled:
                runtime.engage()
                self._sync_enabled = True

            skip_wait = bool((options or {}).get("skip_wait_for_start", False))
            if self.config.wait_for_record_button and not skip_wait:
                observation = self._wait_for_record_start()
                self._recording = True
                info = self._decorate_info(info, event="start", record_reset=True)
            else:
                info = self._decorate_info(info, event=None, record_reset=False)
            if self._recording:
                tqdm.write("[录制中] 正在采集；再次按手柄录制键结束。", file=sys.stderr)
            return observation, info
        except Exception as reset_error:
            try:
                self.close()
            except Exception as cleanup_error:  # pragma: no cover - hardware failure
                self._base_env._logger.error(
                    "YAM leader reset failed (%s), and cleanup also failed: %s",
                    reset_error,
                    cleanup_error,
                )
            raise

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Apply leader, hold, or policy action according to sync configuration."""
        self._ensure_foot_switch()
        if self._review_pending:
            return self._step_review(action)

        runtime = self._base_env.runtime
        try:
            leader_action, buttons = runtime.read_leader_action()
            top_edge, record_edge = self._button_edges(buttons)
            pedal = self._foot_switch.poll() if self._foot_switch is not None else None
            returning = self._return_start is not None
            handoff_hold = False

            if top_edge:
                enabled_now = self._toggle_sync()
                if enabled_now:
                    leader_action, _ = runtime.read_leader_action()
                else:
                    handoff_hold = True
            elif pedal == "reset":
                self._start_return(leader_action)

            returning = returning or self._return_start is not None
            if returning and record_edge:
                tqdm.write(
                    "[复位中] 白键暂不切换录制；复位完成后再按。", file=sys.stderr
                )
                record_edge = False

            if self._return_start is not None:
                effective_action = self._return_action(leader_action)
                action_replaced = True
            elif self._sync_enabled:
                effective_action = leader_action
                action_replaced = True
            elif self.config.unsynced_action_source == "hold" or handoff_hold:
                effective_action = self._base_env.get_hold_action()
                action_replaced = True
            else:
                effective_action = action
                action_replaced = False

            observation, reward, terminated, truncated, info = self.env.step(
                effective_action
            )
            if self._sync_enabled and self._return_start is None:
                runtime.apply_leader_feedback()
            event: str | None = None
            record_reset = False
            manual_done = False
            pending_review = False
            if record_edge:
                if self._recording:
                    if self.config.foot_switch_device is not None:
                        event = "end_pending"
                        reward = 0.0
                        pending_review = True
                    else:
                        event = "end_success"
                        reward = 1.0
                        terminated = True
                        manual_done = True
                else:
                    event = "start"
                    self._recording = True
                    record_reset = True
                    tqdm.write(
                        "[录制中] 正在采集；再次按手柄录制键结束。", file=sys.stderr
                    )
            if (
                self.config.foot_switch_device is not None
                and not record_reset
                and self._recording
                and (terminated or truncated)
            ):
                pending_review = True
                if event is None:
                    event = "end"

            if pending_review:
                terminated = False
                truncated = False
                manual_done = False
                if self._foot_switch is not None:
                    self._foot_switch.begin_review()
                self._review_pending = True

            preserve_manual_sync = bool(
                manual_done
                and not truncated
                and self._sync_enabled
                and self.config.preserve_sync_between_episodes
            )
            self._preserve_sync_on_next_reset = preserve_manual_sync
            # Automatic environment endings still hand ownership back
            # immediately. A configured manual recording boundary preserves
            # synchronization across the collector's following reset.
            if (
                (terminated or truncated)
                and self._sync_enabled
                and not preserve_manual_sync
            ):
                self._disable_sync()
        except Exception:
            self._return_start = None
            self._sync_enabled = False
            self._preserve_sync_on_next_reset = False
            try:
                runtime.emergency_hold()
            except Exception:  # pragma: no cover - hardware failure path
                self._base_env._logger.exception(
                    "Failed to hold YAM followers after a wrapper error"
                )
            try:
                runtime.release_leader_feedback()
            except Exception:  # pragma: no cover - hardware failure path
                self._base_env._logger.exception(
                    "Failed to release YAM leaders after a step error"
                )
            try:
                if self._foot_switch is not None:
                    self._foot_switch.close()
                    self._foot_switch = None
            except Exception:  # pragma: no cover - hardware failure path
                self._base_env._logger.exception(
                    "Failed to close YAM foot switch after a step error"
                )
            raise

        accepted = np.asarray(
            info.get("accepted_action", effective_action), dtype=np.float32
        )
        if action_replaced:
            info["intervene_action"] = accepted
        else:
            info.pop("intervene_action", None)
        info["intervened"] = bool(action_replaced)
        info["manual_done"] = manual_done
        info = self._decorate_info(
            info,
            event=event,
            record_reset=record_reset,
            review_pending=pending_review,
            recording=self._recording,
        )
        if pending_review:
            tqdm.write("[待选择] 左删除，右保存；中不用。", file=sys.stderr)
            self._recording = False
        elif terminated or truncated:
            tqdm.write("[录制结束] 已停止采集；等待后台保存结果。", file=sys.stderr)
        return observation, reward, terminated, truncated, info

    def get_hold_action(self, fallback_action: Any = None) -> np.ndarray:
        """Delegate measured-pose hold generation to the base YAM env."""
        return self._base_env.get_hold_action(fallback_action)

    def close(self) -> None:
        """Release foot-switch input before closing robot resources."""
        first_error: Exception | None = None
        try:
            if self._return_start is not None:
                self._disable_sync()
        except Exception as error:
            first_error = error
        try:
            if self._foot_switch is not None:
                self._foot_switch.close()
        except Exception as error:
            first_error = first_error or error
        finally:
            self._foot_switch = None
        try:
            self.env.close()
        except Exception as error:
            first_error = first_error or error
        if first_error is not None:
            raise first_error

    def _wait_for_record_start(self) -> dict[str, Any]:
        tqdm.write("[待录制] 当前未采集；按手柄录制键开始。", file=sys.stderr)
        period_s = 1.0 / self.config.poll_frequency
        runtime = self._base_env.runtime
        while True:
            started_s = time.perf_counter()
            action, buttons = runtime.read_leader_action()
            top_edge, record_edge = self._button_edges(buttons)
            pedal = self._foot_switch.poll() if self._foot_switch is not None else None
            returning = self._return_start is not None
            if top_edge:
                enabled_now = self._toggle_sync()
                if enabled_now:
                    action, _ = runtime.read_leader_action()
            elif pedal == "reset":
                self._start_return(action)
            returning = returning or self._return_start is not None
            if self._return_start is not None:
                action = self._return_action(action)
            if self._sync_enabled:
                observation, _ = self._base_env.teleop_tick(action)
                if self._return_start is None:
                    runtime.apply_leader_feedback()
            else:
                observation = self._base_env.observe()
            if record_edge and not returning:
                return observation
            remaining_s = period_s - (time.perf_counter() - started_s)
            if remaining_s > 0:
                time.sleep(remaining_s)

    def _step_review(
        self, action: Any
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        del action
        runtime = self._base_env.runtime
        try:
            leader_action, buttons = runtime.read_leader_action()
            top_edge, _ = self._button_edges(buttons)
            if top_edge:
                enabled_now = self._toggle_sync()
                if enabled_now:
                    leader_action, _ = runtime.read_leader_action()

            if self._sync_enabled:
                observation, info = self._base_env.teleop_tick(leader_action)
                runtime.apply_leader_feedback()
            else:
                hold_action = self._base_env.get_hold_action()
                observation = self._base_env.observe()
                info = {"accepted_action": hold_action}

            decision = (
                self._foot_switch.poll() if self._foot_switch is not None else None
            )
            event = None
            manual_done = False
            terminated = False
            episode_discarded = False
            reward = 0.0
            if decision == "keep":
                event = "keep"
                reward = 1.0
                terminated = True
                manual_done = True
                self._review_pending = False
                tqdm.write("[已选择保存] 等待后台保存结果。", file=sys.stderr)
            elif decision == "discard":
                event = "discard"
                terminated = True
                episode_discarded = True
                self._review_pending = False
                tqdm.write("[已选择删除] 当前录制将被丢弃。", file=sys.stderr)

            self._preserve_sync_on_next_reset = bool(
                terminated
                and self._sync_enabled
                and self.config.preserve_sync_between_episodes
            )
            if (
                terminated
                and self._sync_enabled
                and not self._preserve_sync_on_next_reset
            ):
                self._disable_sync()
        except Exception:
            self._sync_enabled = False
            self._preserve_sync_on_next_reset = False
            try:
                runtime.emergency_hold()
            except Exception:  # pragma: no cover - hardware failure path
                self._base_env._logger.exception(
                    "Failed to hold YAM followers after a review error"
                )
            try:
                runtime.release_leader_feedback()
            except Exception:  # pragma: no cover - hardware failure path
                self._base_env._logger.exception(
                    "Failed to release YAM leaders after a review error"
                )
            try:
                if self._foot_switch is not None:
                    self._foot_switch.close()
                    self._foot_switch = None
            except Exception:  # pragma: no cover - hardware failure path
                self._base_env._logger.exception(
                    "Failed to close YAM foot switch after a review error"
                )
            raise

        accepted = np.asarray(
            info.get("accepted_action", leader_action), dtype=np.float32
        )
        info["intervene_action"] = accepted
        info["intervened"] = True
        info["manual_done"] = manual_done
        info["success"] = manual_done
        return (
            observation,
            reward,
            terminated,
            False,
            self._decorate_info(
                info,
                event=event,
                record_reset=False,
                review_pending=self._review_pending,
                episode_discarded=episode_discarded,
                recording=False,
            ),
        )

    def _toggle_sync(self) -> bool:
        runtime = self._base_env.runtime
        if self._sync_enabled:
            self._disable_sync()
        else:
            runtime.engage()
            self._sync_enabled = True
        return self._sync_enabled

    def _disable_sync(self) -> None:
        """Disable software ownership and release feedback even if hold fails."""
        runtime = self._base_env.runtime
        was_returning = self._return_start is not None
        self._return_start = None
        self._sync_enabled = False
        hold_error: Exception | None = None
        try:
            runtime.hold()
        except Exception as error:  # pragma: no cover - hardware failure path
            hold_error = error
        try:
            runtime.release_leader_feedback()
        except Exception as release_error:
            if hold_error is not None:
                raise RuntimeError(
                    "failed to hold followers and release YAM leader feedback"
                ) from release_error
            raise
        if hold_error is not None:
            raise hold_error
        if was_returning:
            tqdm.write("[复位已中止] 从臂保持；主臂恢复重力补偿。", file=sys.stderr)

    def _start_return(self, leader_action: np.ndarray) -> None:
        """Start a joint return without resetting or blocking the episode."""
        if self._return_start is not None:
            return
        if not self._sync_enabled:
            tqdm.write("[无法复位] 请先按黄色键开启跟随。", file=sys.stderr)
            return
        config = self._base_env.config.reset
        if not config.enabled or config.mode != "manual":
            tqdm.write("[无法复位] 尚未配置已确认的数采初始位。", file=sys.stderr)
            return
        target = config.as_vector()
        runtime = self._base_env.runtime
        # The saved leader joint target is also the follower target: normal
        # YAM teleoperation uses absolute joints with no per-arm offset.
        runtime.validate_leader_target(target)
        start = np.stack((runtime.read_state().as_vector(), leader_action.copy()))
        joints = [*range(6), *range(7, 13)]
        limits = np.full(12, config.max_joint_delta)
        if self._base_env.config.enforce_runtime_joint_limits:
            limits = np.minimum(
                limits, np.tile(self._base_env.config.joint_step_limits, 2)
            )
        demand = float(np.max(np.abs(start[:, joints] - target[joints]) / limits))
        self._return_steps = max(
            1,
            math.ceil(config.duration_s * self._base_env.config.step_frequency),
            math.ceil(1.5 * demand),
        )
        self._return_step = 0
        self._return_started_s = time.monotonic()
        self._return_start = start
        phase = "本条继续录制" if self._recording else "当前未录制"
        tqdm.write(
            f"[复位中] 主从双臂返回数采初始位；{phase}。黄色键可中止。",
            file=sys.stderr,
        )

    def _return_action(self, leader_action: np.ndarray) -> np.ndarray:
        """Advance one return frame; the caller records it through env.step."""
        runtime = self._base_env.runtime
        config = self._base_env.config.reset
        target = config.as_vector()
        joints = [*range(6), *range(7, 13)]
        if self._return_step >= self._return_steps:
            measured = runtime.read_state().as_vector()
            error = max(
                float(np.max(np.abs(measured[joints] - target[joints]))),
                float(np.max(np.abs(leader_action[joints] - target[joints]))),
                float(np.max(np.abs(measured[joints] - leader_action[joints]))),
            )
            if error <= config.tolerance:
                runtime.release_leader_feedback()
                self._return_start = None
                suffix = (
                    "本条仍在录制，请按白键结束。"
                    if self._recording
                    else "请按白键开始录制。"
                )
                tqdm.write(f"[复位完成] 主从双臂已到初始位；{suffix}", file=sys.stderr)
                return leader_action
        if time.monotonic() - self._return_started_s >= config.timeout_s:
            self._disable_sync()
            tqdm.write(
                "[复位超时] 已停止回位，请检查姿态；录制状态不变。", file=sys.stderr
            )
            return self._base_env.get_hold_action()
        self._return_step = min(self._return_step + 1, self._return_steps)
        progress = self._return_step / self._return_steps
        alpha = progress * progress * (3.0 - 2.0 * progress)
        actions = self._return_start + alpha * (target - self._return_start)
        # Teaching-handle triggers are passive. Keep grippers under operator
        # control instead of opening automatically while an object is held.
        actions[:, [6, 13]] = leader_action[[6, 13]]
        runtime.command_leaders(actions[1])
        return actions[0]

    def _button_edges(self, buttons: tuple[bool, bool]) -> tuple[bool, bool]:
        now_s = time.monotonic()
        edges_list = []
        for index in range(2):
            rising = bool(buttons[index] and not self._previous_buttons[index])
            accepted = bool(
                rising
                and now_s - self._last_button_edge_s[index]
                >= self.config.button_debounce_s
            )
            if accepted:
                self._last_button_edge_s[index] = now_s
            edges_list.append(accepted)
        self._previous_buttons = buttons
        return edges_list[0], edges_list[1]

    def _ensure_foot_switch(self) -> None:
        if self._foot_switch is not None or self.config.foot_switch_device is None:
            return
        self._foot_switch = FootSwitch(
            self.config.foot_switch_device,
            self.config.foot_switch_discard_key,
            self.config.foot_switch_keep_key,
            reset_key=self.config.foot_switch_reset_key,
        )

    def _decorate_info(
        self,
        info: dict[str, Any],
        *,
        event: str | None,
        record_reset: bool,
        review_pending: bool | None = None,
        episode_discarded: bool = False,
        recording: bool | None = None,
    ) -> dict[str, Any]:
        is_recording = self._recording if recording is None else recording
        is_review_pending = (
            self._review_pending if review_pending is None else review_pending
        )
        phase = "rec" if is_recording else "pre"
        info.update(
            {
                "pre_record": not is_recording,
                "record_reset": bool(record_reset),
                "keyboard_phase": phase,
                "keyboard_event": event,
                "episode_control_phase": phase,
                "episode_control_event": event,
                "episode_review_pending": bool(is_review_pending),
                "episode_discarded": bool(episode_discarded),
                "segment_advance": False,
                "yam_sync_enabled": bool(self._sync_enabled),
            }
        )
        return info

    @property
    def _base_env(self) -> DualYamJointEnv:
        return self.env.unwrapped
