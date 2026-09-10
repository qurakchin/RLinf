#!/usr/bin/env python3
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

"""Safely inspect or replay one RLinf dual-YAM LeRobot parquet episode.

The default invocation is a hardware-free preflight. Passing ``--execute``
connects only the two follower CAN chains; cameras and leader arms are never
opened. Real replay must run on the YAM robot computer.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlinf.envs.real.yam.config import DualYamJointEnvConfig  # noqa: E402
from rlinf.envs.real.yam.control_runtime import YamControlRuntime  # noqa: E402
from rlinf.envs.real.yam.i2rt_backend import (  # noqa: E402
    I2RTYamBackendFactory,
)

_JOINT_INDICES = np.array([*range(6), *range(7, 13)])
_GRIPPER_INDICES = np.array([6, 13])
_JOINT_LIMIT_MIN = [
    [-2.61799, 0.0, 0.0, -1.69297, -1.5708, -2.0944],
    [-2.61799, 0.0, 0.0, -1.69297, -1.5708, -2.0944],
]
_JOINT_LIMIT_MAX = [
    [3.14159, 3.66519, 3.14159, 1.5708, 1.5708, 2.0944],
    [3.14159, 3.66519, 3.14159, 1.5708, 1.5708, 2.0944],
]


@dataclasses.dataclass(frozen=True)
class Episode:
    """Validated trajectory arrays loaded from one parquet file."""

    actions: np.ndarray
    states: np.ndarray | None
    timestamps: np.ndarray

    @property
    def duration_s(self) -> float:
        """Recorded wall-clock duration."""
        return float(self.timestamps[-1] - self.timestamps[0])


def _matrix(table: Any, key: str) -> np.ndarray:
    values = np.asarray(table[key].to_pylist(), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 14:
        raise ValueError(f"{key!r} must have shape (frames, 14), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{key!r} contains non-finite values")
    return values


def load_episode(path: Path, fallback_fps: float) -> Episode:
    """Load canonical actions, optional states, and relative timestamps."""
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError(
            "replay requires pyarrow in the active environment"
        ) from error

    table = pq.read_table(path)
    action_key = next(
        (
            candidate
            for candidate in ("actions", "action")
            if candidate in table.column_names
        ),
        None,
    )
    if action_key is None:
        raise ValueError(f"no action column found; columns={table.column_names}")
    actions = _matrix(table, action_key)

    state_key = next(
        (
            candidate
            for candidate in ("state", "observation.state")
            if candidate in table.column_names
        ),
        None,
    )
    states = None if state_key is None else _matrix(table, state_key)
    if states is not None and len(states) != len(actions):
        raise ValueError("state and action frame counts differ")

    if "timestamp" in table.column_names:
        timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    else:
        timestamps = np.arange(len(actions), dtype=np.float64) / fallback_fps
    if timestamps.shape != (len(actions),) or not np.all(np.isfinite(timestamps)):
        raise ValueError("timestamps must be one finite value per action frame")
    timestamps = timestamps - timestamps[0]
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    return Episode(actions=actions, states=states, timestamps=timestamps)


def validate_episode(
    episode: Episode,
    config: DualYamJointEnvConfig,
    max_recorded_joint_step: float,
) -> None:
    """Reject trajectories outside the configured follower safety envelope."""
    arrays = [("actions", episode.actions)]
    if episode.states is not None:
        arrays.append(("state", episode.states))
    for name, values in arrays:
        for arm_index, offset in enumerate((0, 7)):
            joints = values[:, offset : offset + 6]
            lower = config.joint_limit_min[arm_index]
            upper = config.joint_limit_max[arm_index]
            if np.any(joints < lower) or np.any(joints > upper):
                raise ValueError(
                    f"{name} arm {arm_index} exceeds configured joint limits"
                )
        grippers = values[:, _GRIPPER_INDICES]
        if np.any(grippers < 0.0) or np.any(grippers > 1.0):
            raise ValueError(f"{name} gripper values must stay within [0, 1]")

    if len(episode.actions) > 1:
        largest_step = float(
            np.max(np.abs(np.diff(episode.actions[:, _JOINT_INDICES], axis=0)))
        )
        if largest_step > max_recorded_joint_step:
            raise ValueError(
                f"recorded joint step {largest_step:.4f} exceeds "
                f"--max-recorded-joint-step={max_recorded_joint_step:.4f}"
            )


def print_summary(path: Path, episode: Episode) -> None:
    """Print the motion envelope before any hardware connection."""
    actions = episode.actions
    joint_steps = np.abs(np.diff(actions[:, _JOINT_INDICES], axis=0))
    max_joint_step = float(np.max(joint_steps)) if len(joint_steps) else 0.0
    left_span = np.ptp(actions[:, :7], axis=0)
    right_span = np.ptp(actions[:, 7:], axis=0)
    median_hz = 0.0
    if len(episode.timestamps) > 1:
        median_hz = 1.0 / float(np.median(np.diff(episode.timestamps)))
    print(f"episode: {path}")
    print(f"frames: {len(actions)}")
    print(f"duration: {episode.duration_s:.3f}s")
    print(f"recorded frequency: {median_hz:.3f} Hz")
    print(f"largest recorded arm-joint step: {max_joint_step:.6f} rad")
    print(f"left action span:  {np.round(left_span, 6).tolist()}")
    print(f"right action span: {np.round(right_span, 6).tolist()}")
    print(f"first action: {np.round(actions[0], 6).tolist()}")
    print(f"last action:  {np.round(actions[-1], 6).tolist()}")
    if episode.states is not None:
        print(f"first state:  {np.round(episode.states[0], 6).tolist()}")


def _device(
    channel: str, gripper_limits: list[float] | None, args: argparse.Namespace
) -> SimpleNamespace:
    return SimpleNamespace(
        channel=channel,
        arm_type="yam",
        gripper_type="flexible_4310",
        ee_mass=None,
        gripper_limits=gripper_limits,
        gravity_comp_factor=args.gravity_comp_factor,
        grav_comp_kd=args.grav_comp_kd,
        coulomb_friction=args.coulomb_friction,
        use_coulomb_friction=args.use_coulomb_friction,
        bilateral_kp=0.0,
        gripper_invert=False,
        enable_auto_recovery=False,
    )


def _hardware(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        left_follower=_device(args.left_channel, args.left_gripper_limits, args),
        right_follower=_device(args.right_channel, args.right_gripper_limits, args),
    )


def _wait_until(deadline_s: float) -> None:
    while True:
        remaining_s = deadline_s - time.monotonic()
        if remaining_s <= 0:
            return
        time.sleep(min(remaining_s, 0.01))


def _command_or_raise(runtime: YamControlRuntime, action: np.ndarray) -> bool:
    result = runtime.command(action)
    if result.rejection_reason is not None:
        raise RuntimeError(f"YAM replay command rejected: {result.rejection_reason}")
    return bool(result.clipped)


def _preload_first_action(
    runtime: YamControlRuntime,
    first_action: np.ndarray,
    duration_s: float,
    frequency: float,
) -> int:
    """Slew the commanded setpoint to frame zero without requiring contact convergence."""
    start = runtime.read_state().as_vector()
    steps = max(1, int(np.ceil(duration_s * frequency)))
    period_s = 1.0 / frequency
    clipped = 0
    for step in range(1, steps + 1):
        progress = step / steps
        alpha = progress * progress * (3.0 - 2.0 * progress)
        clipped += _command_or_raise(runtime, start + alpha * (first_action - start))
        if step < steps:
            time.sleep(period_s)
    return clipped


def _hold_target(
    runtime: YamControlRuntime,
    target: np.ndarray,
    frequency: float,
    duration_s: float,
) -> int:
    period_s = 1.0 / frequency
    clipped = 0
    deadline_s = None if duration_s < 0 else time.monotonic() + duration_s
    try:
        while deadline_s is None or time.monotonic() < deadline_s:
            started_s = time.monotonic()
            clipped += _command_or_raise(runtime, target)
            _wait_until(started_s + period_s)
    except KeyboardInterrupt:
        pass
    return clipped


def execute_replay(
    episode: Episode,
    config: DualYamJointEnvConfig,
    args: argparse.Namespace,
) -> None:
    """Own both follower transports until replay and post-hold are complete."""
    runtime = YamControlRuntime(config, _hardware(args), I2RTYamBackendFactory())
    connected = False
    clipped_count = 0
    final_target = episode.actions[-1]
    try:
        runtime.connect_followers()
        connected = True
        runtime.hold()
        if args.move_to_start:
            start_target = (
                episode.states[0] if episode.states is not None else episode.actions[0]
            )
            runtime.move_to(
                start_target,
                duration_s=args.start_duration,
                max_joint_delta=args.start_max_joint_delta,
                tolerance=args.start_tolerance,
                timeout_s=args.start_timeout,
            )

        if not args.yes:
            input(
                "Followers are holding the recorded start state. Prepare the scene, "
                "clear your hands, then press Enter to preload frame 0..."
            )
        clipped_count += _preload_first_action(
            runtime,
            episode.actions[0],
            duration_s=args.preload_duration,
            frequency=config.step_frequency,
        )
        clipped_count += _hold_target(
            runtime,
            episode.actions[0],
            frequency=config.step_frequency,
            duration_s=args.settle_duration,
        )

        print("Starting trajectory replay")
        started_s = time.monotonic()
        for frame_index, (timestamp_s, action) in enumerate(
            zip(episode.timestamps, episode.actions, strict=True)
        ):
            _wait_until(started_s + float(timestamp_s) / args.speed)
            clipped_count += _command_or_raise(runtime, action)
            if frame_index % 30 == 0 or frame_index == len(episode.actions) - 1:
                print(f"replay frame {frame_index + 1}/{len(episode.actions)}")

        print(
            f"Replay complete; {clipped_count} commands were clipped. "
            "Continuing to command the final recorded target."
        )
        if args.hold_after < 0:
            print(
                "Hold is indefinite. Ctrl-C releases the follower transports; "
                "support the arms before stopping."
            )
        clipped_count += _hold_target(
            runtime,
            final_target,
            frequency=config.step_frequency,
            duration_s=args.hold_after,
        )
    except Exception:
        if connected:
            runtime.emergency_hold()
        raise
    finally:
        if connected:
            runtime.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="LeRobot episode parquet")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Connect followers and replay; without this flag only preflight runs.",
    )
    parser.add_argument(
        "--yes", action="store_true", help="Skip interactive confirmations."
    )
    parser.add_argument("--left-channel", default="can_left")
    parser.add_argument("--right-channel", default="can_right")
    parser.add_argument("--left-gripper-limits", type=float, nargs=2, default=None)
    parser.add_argument("--right-gripper-limits", type=float, nargs=2, default=None)
    parser.add_argument(
        "--gravity-comp-factor",
        type=float,
        nargs=6,
        default=None,
        help="Per-joint gravity compensation scale; default keeps the SDK value.",
    )
    parser.add_argument(
        "--grav-comp-kd", type=float, nargs=6, default=None,
        help="Per-joint gravity compensation damping.",
    )
    parser.add_argument(
        "--coulomb-friction", type=float, nargs=6, default=None,
        help="Per-joint coulomb friction compensation; requires "
        "--use-coulomb-friction to take effect.",
    )
    parser.add_argument(
        "--use-coulomb-friction",
        action="store_true",
        help="Enable coulomb friction compensation (technician-tuned stations).",
    )
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--fallback-fps", type=float, default=30.0)
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Replay speed in (0, 1]."
    )
    parser.add_argument("--max-joint-delta", type=float, default=0.05)
    parser.add_argument(
        "--disable-runtime-limits",
        action="store_true",
        help="Match legacy collection by forwarding prevalidated actions directly.",
    )
    parser.add_argument("--max-recorded-joint-step", type=float, default=0.2)
    parser.add_argument(
        "--joint-limit-margin",
        type=float,
        default=0.0,
        help="Expand every joint limit bound by this many radians. Collection "
        "with enforce_runtime_joint_limits=false forwards leader targets "
        "through i2rt's hardware limits, which are slightly wider than the "
        "nominal RLinf limits validated here.",
    )
    parser.add_argument(
        "--move-to-start", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--start-duration", type=float, default=10.0)
    parser.add_argument("--start-max-joint-delta", type=float, default=0.01)
    parser.add_argument("--start-tolerance", type=float, default=0.05)
    parser.add_argument("--start-timeout", type=float, default=40.0)
    parser.add_argument("--preload-duration", type=float, default=3.0)
    parser.add_argument("--settle-duration", type=float, default=1.0)
    parser.add_argument(
        "--hold-after",
        type=float,
        default=-1.0,
        help="Seconds to hold final target; negative means until Ctrl-C.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.episode.is_file():
        raise FileNotFoundError(args.episode)
    positive_values = {
        "frequency": args.frequency,
        "fallback_fps": args.fallback_fps,
        "max_joint_delta": args.max_joint_delta,
        "max_recorded_joint_step": args.max_recorded_joint_step,
        "start_max_joint_delta": args.start_max_joint_delta,
        "start_tolerance": args.start_tolerance,
        "start_timeout": args.start_timeout,
    }
    for name, value in positive_values.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not np.isfinite(args.speed) or not 0.0 < args.speed <= 1.0:
        raise ValueError("--speed must be within (0, 1]")
    for name in ("start_duration", "preload_duration", "settle_duration"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")

    if not np.isfinite(args.joint_limit_margin) or args.joint_limit_margin < 0:
        raise ValueError("--joint-limit-margin must be non-negative")

    limit_min = _JOINT_LIMIT_MIN
    limit_max = _JOINT_LIMIT_MAX
    if args.joint_limit_margin > 0:
        margin = args.joint_limit_margin
        limit_min = [[v - margin for v in arm] for arm in _JOINT_LIMIT_MIN]
        limit_max = [[v + margin for v in arm] for arm in _JOINT_LIMIT_MAX]

    config = DualYamJointEnvConfig(
        step_frequency=args.frequency,
        max_joint_delta=args.max_joint_delta,
        enforce_runtime_joint_limits=not args.disable_runtime_limits,
        joint_limit_min=limit_min,
        joint_limit_max=limit_max,
    )
    episode = load_episode(args.episode, args.fallback_fps)
    validate_episode(episode, config, args.max_recorded_joint_step)
    print_summary(args.episode, episode)
    if not args.execute:
        print("Preflight passed. No hardware was opened. Add --execute on yambox.")
        return
    if not args.yes:
        answer = input(
            "This will move both follower arms and both grippers. Clear the "
            "workspace and type REPLAY to continue: "
        )
        if answer != "REPLAY":
            print("Replay cancelled; no hardware was opened.")
            return
    execute_replay(episode, config, args)


if __name__ == "__main__":
    main()
