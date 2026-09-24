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

"""Capture wrist-camera images and measured joint states for hand-eye calibration.

Drives one YAM follower through small joint-space perturbations around its
current pose (the other arm is held in place), capturing one wrist-camera
image per pose. Poses are kept small and slow; the standard runtime safety
layer (limit clipping, feedback checks) stays active.

Output directory contents:
    images/pose_XXX.png     one wrist-camera frame per reached pose
    poses.npz               measured 14-D states, one row per image
    meta.json               camera serial/intrinsics and arm side
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

_JOINT_LIMIT_MIN = np.array([-2.61799, 0.0, 0.0, -1.69297, -1.5708, -2.0944])
_JOINT_LIMIT_MAX = np.array([3.14159, 3.66519, 3.14159, 1.5708, 1.5708, 2.0944])

# Wrist/elbow-focused perturbations (radians) around the measured pose.
# Rotational diversity is what calibrateHandEye needs; translations come from
# the elbow/shoulder rows.
_PERTURBATIONS = [
    (0, {}),
    (1, {3: 0.15}),
    (2, {3: -0.15}),
    (3, {4: 0.20}),
    (4, {4: -0.20}),
    (5, {5: 0.20}),
    (6, {5: -0.20}),
    (7, {4: 0.20, 5: 0.20}),
    (8, {4: 0.20, 5: -0.20}),
    (9, {4: -0.20, 5: 0.20}),
    (10, {4: -0.20, 5: -0.20}),
    (11, {2: 0.12, 4: 0.15}),
    (12, {2: -0.12, 4: -0.15}),
    (13, {1: 0.08, 3: 0.15, 5: 0.15}),
    (14, {1: -0.08, 3: -0.15, 5: -0.15}),
    (15, {0: 0.10, 4: 0.20}),
    (16, {0: -0.10, 4: -0.20}),
]


def _device(channel: str) -> SimpleNamespace:
    # Matches the station's technician-tuned gains (see collect yaml).
    return SimpleNamespace(
        channel=channel,
        arm_type="yam",
        gripper_type="flexible_4310",
        ee_mass=None,
        gripper_limits=None,
        gravity_comp_factor=[1.0, 1.1, 1.1, 1.2, 1.0, 1.0],
        grav_comp_kd=None,
        coulomb_friction=None,
        use_coulomb_friction=True,
        bilateral_kp=0.0,
        gripper_invert=False,
        # Motors latch a "loss communication" error after sitting idle; let the
        # driver clear and re-enable them instead of failing the connect.
        enable_auto_recovery=True,
    )


class _Camera:
    def __init__(self, serial: str):
        import pyrealsense2 as rs

        self._rs = rs
        self._pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self._pipe.start(cfg)
        intr = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self.intrinsics = {
            "model": str(intr.model),
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.ppx,
            "cy": intr.ppy,
            "distortion": list(intr.coeffs),
        }
        for _ in range(15):  # let auto-exposure settle
            self._pipe.wait_for_frames()

    def capture(self) -> np.ndarray:
        frames = self._pipe.wait_for_frames()
        return np.asanyarray(frames.get_color_frame().get_data())

    def close(self) -> None:
        self._pipe.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("left", "right"), required=True)
    parser.add_argument(
        "--camera",
        choices=tuple(CAMERA_SERIALS),
        default=None,
        help="Camera to capture from; defaults to the arm's wrist camera.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Multiplier on all perturbation magnitudes.",
    )
    parser.add_argument(
        "--extra-offset",
        type=float,
        nargs=6,
        default=None,
        metavar=("J0", "J1", "J2", "J3", "J4", "J5"),
        help="Constant joint offset added to every pose, e.g. to "
        "turn the arm toward the board first.",
    )
    parser.add_argument("--settle-s", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.scale <= 1.5:
        raise ValueError("--scale must be within (0, 1.5]")

    import cv2

    camera_name = args.camera or args.arm
    serial = CAMERA_SERIALS[camera_name]
    out_images = args.out / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    cfg = DualYamJointEnvConfig(
        step_frequency=30.0,
        max_joint_delta=0.05,
        enforce_runtime_joint_limits=True,
        joint_limit_min=[_JOINT_LIMIT_MIN.tolist()] * 2,
        joint_limit_max=[_JOINT_LIMIT_MAX.tolist()] * 2,
    )
    runtime = YamControlRuntime(
        cfg,
        SimpleNamespace(
            left_follower=_device("can_left"), right_follower=_device("can_right")
        ),
        I2RTYamBackendFactory(),
    )

    arm_offset = 0 if args.arm == "left" else 7
    captured: list[dict] = []
    camera = None
    try:
        runtime.connect_followers()
        runtime.hold()
        camera = _Camera(serial)

        base = runtime.read_state().as_vector()
        extra = np.zeros(6)
        if args.extra_offset is not None:
            extra = np.asarray(args.extra_offset, dtype=np.float64)

        for pose_idx, deltas in _PERTURBATIONS:
            target = base.copy()
            joints = base[arm_offset : arm_offset + 6] + extra
            for j, d in deltas.items():
                joints[j] += d * args.scale
            joints = np.clip(joints, _JOINT_LIMIT_MIN + 0.02, _JOINT_LIMIT_MAX - 0.02)
            target[arm_offset : arm_offset + 6] = joints
            # Hold the other arm and both grippers at their measured state.
            try:
                runtime.move_to(
                    target,
                    duration_s=2.0,
                    max_joint_delta=0.02,
                    tolerance=0.03,
                    timeout_s=12.0,
                )
            except Exception as error:
                print(f"pose {pose_idx}: move failed ({error}); skipping")
                runtime.hold()
                continue
            time.sleep(args.settle_s)
            measured = runtime.read_state().as_vector()
            image = camera.capture()
            path = out_images / f"pose_{len(captured):03d}.png"
            cv2.imwrite(str(path), image)
            captured.append({"image": path.name, "measured": measured.tolist()})
            print(f"pose {pose_idx} -> {path.name}")
    finally:
        if camera is not None:
            camera.close()
        runtime.close()

    if captured:
        np.savez(
            args.out / "poses.npz",
            measured=np.array([c["measured"] for c in captured]),
            images=np.array([c["image"] for c in captured]),
        )
        (args.out / "meta.json").write_text(
            json.dumps(
                {
                    "arm": args.arm,
                    "camera": camera_name,
                    "camera_serial": serial,
                    "intrinsics": camera.intrinsics if camera else None,
                    "num_poses": len(captured),
                },
                indent=2,
            )
        )
    print(f"captured {len(captured)} poses -> {args.out}")


if __name__ == "__main__":
    main()
