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

"""Standalone calibration data collection for the dual-YAM station.

Both follower arms are connected in zero-gravity mode (via the proven RLinf
control-runtime connect path): the operator pushes them by hand to point the
wrist cameras at the calibration board. The script never commands motion
during collection. It records a sample (3 camera images + measured 14-D joint
state) automatically whenever the arms have been stationary for a moment and
the pose differs enough from the last saved sample, which yields sharp,
diverse viewpoints for hand-eye calibration.

Output directory contents:
    <camera>/sample_XXXXXX.png   one image per camera per sample
    frames.jsonl                 one JSON line per sample with the 14-D state
    meta.json                    camera intrinsics and sampling parameters

Stop with Ctrl-C. Every sample is flushed to disk immediately, so nothing is
lost on exit. On exit the arms are smoothly driven back to their startup pose
before torque is cut, so they never drop freely (use --no-park to skip).

Safety note: arms become compliant (gravity-compensated) as soon as the
script connects, and both grippers run their auto-calibration sweep at
startup. Support the arms while starting and stopping.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rlinf.envs.real.yam.config import DualYamJointEnvConfig  # noqa: E402
from rlinf.envs.real.yam.control_runtime import YamControlRuntime  # noqa: E402
from rlinf.envs.real.yam.i2rt_backend import I2RTYamBackendFactory  # noqa: E402

CAMERA_SERIALS = {
    "top": "260322277483",
    "left": "260322272602",
    "right": "260422273719",
}

_LOOP_PERIOD_S = 1.0 / 30.0

_JOINT_LIMIT_MIN = [-2.61799, 0.0, 0.0, -1.69297, -1.5708, -2.0944]
_JOINT_LIMIT_MAX = [3.14159, 3.66519, 3.14159, 1.5708, 1.5708, 2.0944]


def _clear_motor_errors(channels=("can_left", "can_right")) -> None:
    """Clean latched DM-motor errors (e.g. idle watchdog 'loss communication').

    The motors latch an error whenever a previous session ends; i2rt's startup
    checks then fail before its own clean/enable loop gets control. Sending the
    raw clean-error (0xFB) + enable (0xFC) frames right before connecting keeps
    the motors fresh through the startup window.
    """
    import can

    for channel in channels:
        bus = can.interface.Bus(channel=channel, interface="socketcan")
        try:
            while bus.recv(timeout=0.01):
                pass
            for mid in range(1, 8):
                clean = can.Message(
                    arbitration_id=mid,
                    data=[0xFF] * 7 + [0xFB],
                    is_extended_id=False,
                )
                for _ in range(3):
                    bus.send(clean)
                    time.sleep(0.005)
                time.sleep(0.02)
                while bus.recv(timeout=0.005):
                    pass
                bus.send(
                    can.Message(
                        arbitration_id=mid,
                        data=[0xFF] * 7 + [0xFC],
                        is_extended_id=False,
                    )
                )
                time.sleep(0.02)
        finally:
            bus.shutdown()


def _device(channel: str) -> SimpleNamespace:
    # Zero-gravity float for hand-guiding: coulomb friction compensation is a
    # position-tracking aid and can self-drive a floating arm, so keep it off
    # here even though the station's tuned collection config enables it.
    return SimpleNamespace(
        channel=channel,
        arm_type="yam",
        gripper_type="flexible_4310",
        ee_mass=None,
        gripper_limits=None,
        gravity_comp_factor=[1.0, 1.1, 1.1, 1.2, 1.0, 1.0],
        grav_comp_kd=None,
        coulomb_friction=None,
        use_coulomb_friction=False,
        bilateral_kp=0.0,
        gripper_invert=False,
        enable_auto_recovery=True,
        zero_gravity_mode=True,
    )


def _float_arms(runtime: YamControlRuntime) -> None:
    """Switch both followers from PD hold to gravity-comp idle (hand-guidable).

    ``connect_followers()`` holds each arm at its measured pose so the startup
    sequence is safe; call this only after cameras are up and the operator is
    supporting the arms.
    """
    followers = getattr(runtime, "_followers", None)
    if not followers:
        raise RuntimeError("followers are not connected")
    for backend in followers:
        robot = getattr(backend, "_robot", None)
        if robot is None or not hasattr(robot, "enter_gravity_comp_idle"):
            raise RuntimeError("follower backend cannot enter gravity-comp idle")
        robot.enter_gravity_comp_idle()


class _Camera:
    def __init__(self, serial: str, width: int, height: int, fps: int):
        import pyrealsense2 as rs

        self._rs = rs
        self._pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        # Arm startup saturates CAN/CPU and can starve the first frames;
        # retry rather than abort the whole session.
        last_error = None
        for _ in range(3):
            try:
                profile = self._pipe.start(cfg)
                for _ in range(15):  # let auto-exposure settle
                    self._pipe.wait_for_frames()
                break
            except RuntimeError as error:
                last_error = error
                try:
                    self._pipe.stop()
                except Exception:
                    pass
                self._pipe = rs.pipeline()
                time.sleep(1.0)
        else:
            raise RuntimeError(f"camera {serial} failed to start: {last_error}")
        intr = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self.intrinsics = {
            "model": str(intr.model),
            "width": intr.width,
            "height": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.ppx,
            "cy": intr.ppy,
            "distortion": list(intr.coeffs),
        }
        self._latest = None

    def poll(self) -> None:
        frames = self._pipe.poll_for_frames()
        if frames:
            self._latest = np.asanyarray(frames.get_color_frame().get_data())

    @property
    def latest(self) -> np.ndarray | None:
        return self._latest

    def close(self) -> None:
        self._pipe.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--stationary-window-s",
        type=float,
        default=0.4,
        help="Arm must stay within --stationary-tol for this long to trigger.",
    )
    parser.add_argument(
        "--stationary-tol",
        type=float,
        default=0.004,
        help="Max per-joint motion (rad) inside the stationary window.",
    )
    parser.add_argument(
        "--min-pose-distance",
        type=float,
        default=0.08,
        help="Min max-joint displacement (rad) from the last saved sample.",
    )
    # All three cameras share limited USB bandwidth; 640x480 bgr8 @30fps is
    # the highest triple-camera profile the station can sustain (verified on
    # the YAM Box). The RLinf collection uses the same resolution.
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--no-park",
        action="store_true",
        help="Skip driving the arms back to their startup pose on exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import cv2

    args.out.mkdir(parents=True, exist_ok=True)
    for name in CAMERA_SERIALS:
        (args.out / name).mkdir(exist_ok=True)

    cfg = DualYamJointEnvConfig(
        step_frequency=30.0,
        max_joint_delta=0.05,
        enforce_runtime_joint_limits=True,
        joint_limit_min=[_JOINT_LIMIT_MIN] * 2,
        joint_limit_max=[_JOINT_LIMIT_MAX] * 2,
    )
    runtime = YamControlRuntime(
        cfg,
        SimpleNamespace(
            left_follower=_device("can_left"), right_follower=_device("can_right")
        ),
        I2RTYamBackendFactory(),
    )

    cameras = {}
    frames_fp = open(args.out / "frames.jsonl", "a")
    count = 0
    start_state = None
    try:
        _clear_motor_errors()
        runtime.connect_followers()
        # Arms stay in PD hold while the cameras open, then switch to
        # gravity-comp idle for hand-guiding.
        start_state = runtime.read_state().as_vector()
        # Let the arm control loops settle before opening the USB cameras.
        time.sleep(1.0)
        for name, serial in CAMERA_SERIALS.items():
            cameras[name] = _Camera(serial, args.width, args.height, 30)
        print("Arms are held. Support them now; they become compliant in 3 s...")
        time.sleep(3.0)
        _float_arms(runtime)

        (args.out / "meta.json").write_text(
            json.dumps(
                {
                    "cameras": {
                        name: {
                            "serial": CAMERA_SERIALS[name],
                            **cam.intrinsics,
                        }
                        for name, cam in cameras.items()
                    },
                    "state_layout": "[left q0..q5, left_gripper_01, "
                    "right q0..q5, right_gripper_01] (measured; gripper "
                    "normalized 0=closed, 1=open)",
                    "stationary_window_s": args.stationary_window_s,
                    "stationary_tol": args.stationary_tol,
                    "min_pose_distance": args.min_pose_distance,
                },
                indent=2,
            )
        )

        window: list[np.ndarray] = []
        last_saved: np.ndarray | None = None
        print(
            "Zero-gravity active. Push the arms by hand; samples are taken "
            "automatically when an arm holds still at a new pose. Ctrl-C to stop."
        )
        while True:
            started = time.monotonic()
            for cam in cameras.values():
                cam.poll()
            state = runtime.read_state().as_vector()

            window.append(state)
            max_len = max(2, int(args.stationary_window_s / _LOOP_PERIOD_S))
            if len(window) > max_len:
                window.pop(0)
            spread = float(np.max(np.ptp(np.asarray(window), axis=0)))
            stationary = len(window) == max_len and spread < args.stationary_tol
            diverse = (
                last_saved is None
                or float(np.max(np.abs(state - last_saved))) > args.min_pose_distance
            )

            if stationary and diverse:
                images = {name: cam.latest for name, cam in cameras.items()}
                if all(img is not None for img in images.values()):
                    for name, img in images.items():
                        cv2.imwrite(
                            str(args.out / name / f"sample_{count:06d}.png"), img
                        )
                    frames_fp.write(
                        json.dumps({"sample": count, "state": state.tolist()}) + "\n"
                    )
                    frames_fp.flush()
                    last_saved = state
                    count += 1
                    print(f"sample {count} saved", flush=True)

            elapsed = time.monotonic() - started
            if elapsed < _LOOP_PERIOD_S:
                time.sleep(_LOOP_PERIOD_S - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        frames_fp.close()
        for cam in cameras.values():
            cam.close()
        if start_state is not None and not args.no_park:
            print("Parking arms at their startup pose before torque-off...")
            try:
                runtime.move_to(
                    start_state,
                    duration_s=3.0,
                    max_joint_delta=0.02,
                    tolerance=0.05,
                    timeout_s=12.0,
                )
            except Exception as error:
                print(f"park failed ({error}); support the arms before closing")
        runtime.close()
    print(f"done: {count} samples -> {args.out}")


if __name__ == "__main__":
    main()
