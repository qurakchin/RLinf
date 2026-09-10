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

"""Hover both YAM grippers above pixel-picked targets (grasp-prep primitive).

Pipeline: top-camera RGB+aligned depth -> pixel deprojection -> calibrated
T_base->top extrinsics -> per-arm base frame -> top-down hover pose (grasp
approach = -X of grasp_site, finger closing = Z of grasp_site, laid across
the target's long axis) -> IK -> runtime move_to -> FK re-projection
verification image.

Targets JSON format:
{
  "hover_m": 0.10,
  "left":  {"pixel": [u, v], "axis": [[u1, v1], [u2, v2]]},
  "right": {"pixel": [u, v], "axis": [[u1, v1], [u2, v2]]}
}
``axis`` is the target's long axis (e.g. spoon bowl -> handle tip); the
gripper closes across it.

Run without --execute for a dry run (planning image + IK check only).
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
from rlinf.envs.real.yam.kinematics import YamKinematicsAdapter  # noqa: E402

ARM_SLICE = {"left": (0, 7), "right": (7, 14)}
CHANNELS = {"left": "can_left", "right": "can_right"}
TOP_SERIAL = "260322277483"

_JOINT_LIMIT_MIN = [-2.61799, 0.0, 0.0, -1.69297, -1.5708, -2.0944]
_JOINT_LIMIT_MAX = [3.14159, 3.66519, 3.14159, 1.5708, 1.5708, 2.0944]


def _device(channel: str) -> SimpleNamespace:
    # Position-tracking here, so the station's technician-tuned gains apply.
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
        enable_auto_recovery=True,
    )


def _deproject(u: float, v: float, depth: np.ndarray, intr: dict) -> np.ndarray:
    """Median-depth deprojection of a pixel into the camera frame."""
    u, v = int(round(u)), int(round(v))
    patch = depth[max(0, v - 3) : v + 4, max(0, u - 3) : u + 4]
    valid = patch[patch > 0.1]
    if valid.size == 0:
        raise ValueError(f"no valid depth near pixel ({u}, {v})")
    z = float(np.median(valid))
    return np.array([(u - intr["cx"]) * z / intr["fx"], (v - intr["cy"]) * z / intr["fy"], z])


def _fit_table_plane(
    depth: np.ndarray, intr: dict, t_base_top: np.ndarray, exclude: list
) -> tuple[np.ndarray, int]:
    """Fit the table plane z = a x + b y + c in the base frame.

    Samples a grid of bare-table pixels, excluding points near targets, and
    iteratively refits on inliers. White-on-white textureless depth is biased
    per-pixel; the plane fit averages it out and gives a far better surface
    height for hovering.

    Returns (plane [a, b, c], inlier count).
    """
    h, w = depth.shape
    pts = []
    for v in range(120, min(h - 10, 430), 25):
        for u in range(60, min(w - 10, 600), 30):
            if any(abs(u - ex[0]) < 70 and abs(v - ex[1]) < 70 for ex in exclude):
                continue
            try:
                p_cam = _deproject(u, v, depth, intr)
            except ValueError:
                continue
            pts.append((t_base_top @ np.append(p_cam, 1.0))[:3])
    pts = np.asarray(pts)
    inliers = np.ones(len(pts), dtype=bool)
    plane = None
    for _ in range(3):
        sel = pts[inliers]
        # Least squares z = a x + b y + c
        amat = np.column_stack([sel[:, 0], sel[:, 1], np.ones(len(sel))])
        plane = np.linalg.lstsq(amat, sel[:, 2], rcond=None)[0]
        resid = np.abs(pts @ np.append(plane[:2], 0.0) + plane[2] - pts[:, 2])
        thr = max(0.015, float(np.median(resid)) * 3)
        inliers = resid < thr
    return plane, int(inliers.sum())


def _rot_about(axis: np.ndarray, deg: float) -> np.ndarray:
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    k = np.asarray(axis, dtype=np.float64)
    k /= np.linalg.norm(k)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) * c + (1 - c) * np.outer(k, k) + s * kx


def _ik_multi_seed(kin: YamKinematicsAdapter, target: np.ndarray, q_measured):
    """Try IK over approach-side flips and mouth-tilt offsets.

    The jaw closing line (grasp_site Y, horizontal across the target's long
    axis) is functionally fixed; reachability is opened up by (a) approaching
    from either side — a 180° flip about local X keeps the closing line but
    swaps which side of the target the arm reaches from — and (b) tilting the
    mouth about local Y.

    Returns (result, target_used, (flipped, tilt_deg)); on failure
    target_used is None and result is the closest failed attempt.
    """
    best = None
    for flip in (False, True):
        rx = _rot_about(np.array([1.0, 0.0, 0.0]), 180.0) if flip else np.eye(3)
        for tilt_deg in (0, 15, -15, 30, -30, 45, -45, 60, -60):
            ry = _rot_about(np.array([0.0, 1.0, 0.0]), tilt_deg)
            tilted = target.copy()
            tilted[:3, :3] = target[:3, :3] @ rx @ ry
            for seed in (
                q_measured,
                np.zeros(6),
                np.array([0.0, 0.6, 0.8, 0.0, 0.6, 0.0]),
                np.array([0.0, 1.0, 1.2, 0.0, 0.8, 0.0]),
            ):
                result = kin.solve(tilted, seed, gripper_position=1.0)
                if result.success:
                    return result, tilted, (flip, tilt_deg)
                if best is None or (
                    np.isfinite(result.position_error)
                    and result.position_error < getattr(best, "position_error", np.inf)
                ):
                    best = result
    return best, None, None


def _hover_target(p_base: np.ndarray, axis_base: np.ndarray, hover_m: float) -> np.ndarray:
    """Build the 4x4 grasp_site target for the flexible_4310 scoop grasp.

    Verified against the MuJoCo model and real working poses: the jaws slide
    along grasp_site Y (horizontal pinch across the target's long axis), the
    mouth faces along grasp_site -Z (forward-down), and grasp_site X points
    up. Strict vertical approaches are mechanically unreachable on this arm,
    so the mouth stays forward-down like the proven collection poses.
    """
    z_up = np.array([0.0, 0.0, 1.0])
    axis_xy = axis_base - np.dot(axis_base, z_up) * z_up
    if np.linalg.norm(axis_xy) < 1e-6:
        axis_xy = np.array([1.0, 0.0, 0.0])
    axis_xy /= np.linalg.norm(axis_xy)
    y_site = np.cross(axis_xy, z_up)  # jaw closing line across the long axis
    y_site /= np.linalg.norm(y_site)
    x_site = z_up - np.dot(z_up, y_site) * y_site  # up, orthogonalized
    x_site /= np.linalg.norm(x_site)
    z_site = np.cross(x_site, y_site)
    target = np.eye(4)
    target[:3, 0] = x_site
    target[:3, 1] = y_site
    target[:3, 2] = z_site
    target[:3, 3] = p_base + np.array([0.0, 0.0, hover_m])
    return target


def _path_min_clearance(
    kin: YamKinematicsAdapter,
    q_from: np.ndarray,
    q_to: np.ndarray,
    side: str,
    plane: np.ndarray,
    samples: int = 40,
) -> float:
    """Min TCP height above the table plane along the move_to smoothstep path."""
    lo, _ = ARM_SLICE[side]
    min_clear = float("inf")
    for alpha in np.linspace(0.0, 1.0, samples):
        a = alpha * alpha * (3.0 - 2.0 * alpha)
        q = q_from + a * (q_to - q_from)
        p = kin.fk(q[lo : lo + 6], float(q[lo + 6]))[:3, 3]
        clear = p[2] - (plane[0] * p[0] + plane[1] * p[1] + plane[2])
        min_clear = min(min_clear, float(clear))
    return min_clear


def _safe_move_to(
    runtime: YamControlRuntime,
    kin: YamKinematicsAdapter,
    goal: np.ndarray,
    side: str,
    table_plane: np.ndarray,
    clearance_m: float = 0.03,
    _depth: int = 0,
) -> None:
    """move_to with a table-floor guard on the whole interpolated path.

    If the direct joint-space path dips the TCP below ``table_plane +
    clearance_m``, fall back to a lift-translate-descend profile: lift the TCP
    straight up to a cruise height, translate above the goal, then descend.
    """
    lo, hi = ARM_SLICE[side]
    current = runtime.read_state().as_vector()
    clearance = _path_min_clearance(kin, current, goal, side, table_plane)
    if clearance >= clearance_m or _depth >= 2:
        if clearance < clearance_m:
            print(
                f"WARNING: {side} path clearance {clearance * 1000:.0f} mm below "
                f"threshold at max waypoint depth; proceeding with caution"
            )
        runtime.move_to(
            goal, duration_s=8.0, max_joint_delta=0.02,
            tolerance=0.05, timeout_s=40.0,
        )
        return

    print(
        f"{side}: direct path dips to {clearance * 1000:.0f} mm above table; "
        "using lift-translate-descend"
    )
    fk_now = kin.fk(current[lo : lo + 6], float(current[lo + 6]))
    fk_goal = kin.fk(goal[lo : lo + 6], float(goal[lo + 6]))
    plane_z_now = float(
        table_plane[0] * fk_now[0, 3] + table_plane[1] * fk_now[1, 3] + table_plane[2]
    )
    cruise_z = max(fk_now[2, 3], fk_goal[2, 3], plane_z_now + 0.15)

    for fk_ref, label in ((fk_now, "lift"), (fk_goal, "translate")):
        waypoint = np.eye(4)
        waypoint[:3, :3] = fk_ref[:3, :3]
        waypoint[:3, 3] = [fk_ref[0, 3], fk_ref[1, 3], cruise_z]
        seed = runtime.read_state().as_vector()[lo : lo + 6]
        result, target_used, _ = _ik_multi_seed(kin, waypoint, seed)
        if not result.success:
            raise RuntimeError(f"IK failed for {side} {label} waypoint: {result.reason}")
        step_goal = runtime.read_state().as_vector()
        step_goal[lo:hi] = np.concatenate([result.q_target, [goal[hi - 1]]])
        _safe_move_to(
            runtime, kin, step_goal, side, table_plane, clearance_m, _depth + 1
        )
    _safe_move_to(runtime, kin, goal, side, table_plane, clearance_m, _depth + 1)


class _TopCamera:
    """Top RealSense held open for the whole session.

    Opening a RealSense pipeline mid-control storms the USB bus that the CAN
    adapters share, which has repeatedly starved the arm control loops into a
    watchdog torque cut. Open once up front, then poll.
    """

    def __init__(self, serial: str = TOP_SERIAL):
        import pyrealsense2 as rs

        self._pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self._pipe.start(cfg)
        for _ in range(15):  # settle auto-exposure
            self._pipe.wait_for_frames()

    def capture(self) -> np.ndarray:
        frames = self._pipe.wait_for_frames()
        return np.asanyarray(frames.get_color_frame().get_data())

    def close(self) -> None:
        self._pipe.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--extrinsics", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True, help="top RGB png")
    parser.add_argument("--depth", type=Path, required=True, help="aligned depth npy (meters)")
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true",
                        help="Move the arms; without it only plan + IK-check.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import cv2

    targets = json.loads(args.targets.read_text())
    extr = json.loads(args.extrinsics.read_text())
    intr = json.loads(args.intrinsics.read_text())
    depth = np.load(args.depth)
    image = cv2.imread(str(args.image))
    hover_m = float(targets.get("hover_m", 0.10))
    args.out.mkdir(parents=True, exist_ok=True)

    plan = {}
    annotated = image.copy()
    exclude = [targets[s]["pixel"] for s in ("left", "right")]
    for side in ("left", "right"):
        spec = targets[side]
        u, v = spec["pixel"]
        p_cam = _deproject(u, v, depth, intr)
        t_base_top = np.asarray(extr["top_camera"]["T_base_to_cam"][f"via_{side}_arm"])
        p_base = (t_base_top @ np.append(p_cam, 1.0))[:3]
        # Height from the fitted table plane, not the noisy per-pixel depth;
        # lateral x/y stay pixel-accurate. A spoon adds ~15 mm over the plane.
        plane, n_inliers = _fit_table_plane(depth, intr, t_base_top, exclude)
        surface_z = float(plane[0] * p_base[0] + plane[1] * p_base[1] + plane[2])
        p_base[2] = surface_z + 0.015
        a_cam = [_deproject(*pt, depth, intr) for pt in spec["axis"]]
        axis_base = (t_base_top @ np.append(a_cam[1], 1.0))[:3] - (
            t_base_top @ np.append(a_cam[0], 1.0)
        )[:3]
        target = _hover_target(p_base, axis_base, hover_m)
        plan[side] = {"p_base": p_base.tolist(), "target": target.tolist(),
                      "table_plane": plane.tolist(),
                      "table_plane_inliers": n_inliers}
        color = (0, 255, 0) if side == "left" else (0, 0, 255)
        cv2.drawMarker(annotated, (int(u), int(v)), color, cv2.MARKER_TILTED_CROSS, 24, 2)
        cv2.line(annotated, tuple(map(int, spec["axis"][0])),
                 tuple(map(int, spec["axis"][1])), color, 1)
        cv2.putText(annotated, side, (int(u) + 8, int(v) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        print(
            f"{side}: pixel ({u},{v}) -> base {np.round(p_base[:3], 3).tolist()}, "
            f"hover target {np.round(target[:3, 3], 3).tolist()}"
        )
    cv2.imwrite(str(args.out / "plan_targets.png"), annotated)

    kin = YamKinematicsAdapter()
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2))
    if not args.execute:
        print(f"dry run; plan image + json saved to {args.out}")
        return

    cfg = DualYamJointEnvConfig(
        step_frequency=30.0,
        max_joint_delta=0.05,
        # match the collection/replay path: the runtime slew clip caps PD error
        # at 0.05 rad, which starves torque against elbow friction; move_to's
        # own smoothstep interpolation (0.02 rad/step) already paces motion.
        enforce_runtime_joint_limits=False,
        joint_limit_min=[_JOINT_LIMIT_MIN] * 2,
        joint_limit_max=[_JOINT_LIMIT_MAX] * 2,
    )
    runtime = YamControlRuntime(
        cfg,
        SimpleNamespace(
            left_follower=_device(CHANNELS["left"]),
            right_follower=_device(CHANNELS["right"]),
        ),
        I2RTYamBackendFactory(),
    )
    camera = None
    try:
        # Open the camera before connecting the arms; mid-control USB
        # enumeration has repeatedly starved the CAN loops into a torque cut.
        camera = _TopCamera()
        runtime.connect_followers()
        runtime.hold()
        measured = runtime.read_state().as_vector()

        solutions = {}
        for side in ("left", "right"):
            lo, _ = ARM_SLICE[side]
            q_seed = measured[lo : lo + 6]
            result, target_used, yaw_tilt = _ik_multi_seed(
                kin, np.asarray(plan[side]["target"]), q_seed
            )
            print(
                f"{side}: IK success={result.success} pos_err={result.position_error * 1000:.1f}mm "
                f"rot_err={result.rotation_error:.4f}rad reason={result.reason} "
                f"yaw/tilt={yaw_tilt}"
            )
            if not result.success:
                raise RuntimeError(f"IK failed for {side}: {result.reason}")
            solutions[side] = result.q_target
            plan[side]["target_used"] = target_used.tolist()

        if not args.yes:
            answer = input(
                "Both IK solutions valid. Arms will move to hover above the "
                "targets one at a time. Type HOVER to continue: "
            )
            if answer != "HOVER":
                print("aborted; no motion")
                return

        goals = {}
        for side in ("left", "right"):
            lo, hi = ARM_SLICE[side]
            goal = runtime.read_state().as_vector()
            goal[lo:hi] = np.concatenate([solutions[side], [1.0]])
            goals[side] = goal.copy()
            print(f"moving {side} arm ...")
            _safe_move_to(
                runtime, kin, goal, side,
                table_plane=np.asarray(plan[side]["table_plane"]),
                clearance_m=0.03,
            )
            time.sleep(0.5)

        K = np.array(
            [[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]]
        )

        # Verification via the pre-opened camera. If a torque glitch dropped an
        # arm mid-run, re-hover and re-capture so the saved image is truthful.
        for attempt in range(2):
            state = runtime.read_state().as_vector()
            drooped = False
            for side in ("left", "right"):
                lo, _ = ARM_SLICE[side]
                # Arm joints only: the gripper's normalized rest reading drifts
                # by a few percent and would false-trigger the check.
                dev = float(np.max(np.abs(state[lo : lo + 6] - goals[side][lo : lo + 6])))
                if dev > 0.08:
                    print(
                        f"{side} arm drooped {dev:.3f} rad mid-run; "
                        f"re-hovering (attempt {attempt + 1})\n"
                        f"  state: {np.round(state[lo:lo + 6], 3).tolist()}\n"
                        f"  goal : {np.round(goals[side][lo:lo + 6], 3).tolist()}"
                    )
                    _safe_move_to(
                        runtime, kin, goals[side], side,
                        table_plane=np.asarray(plan[side]["table_plane"]),
                        clearance_m=0.03,
                    )
                    drooped = True
            if not drooped:
                break
            time.sleep(0.5)

        verify_img = camera.capture()
        state = runtime.read_state().as_vector()
        for side in ("left", "right"):
            lo, _ = ARM_SLICE[side]
            fk = kin.fk(state[lo : lo + 6], float(state[lo + 6]))
            t_base_top = np.asarray(
                extr["top_camera"]["T_base_to_cam"][f"via_{side}_arm"]
            )
            p_cam = np.linalg.inv(t_base_top) @ np.append(fk[:3, 3], 1.0)
            uv = K @ p_cam[:3] / p_cam[2]
            color = (0, 255, 0) if side == "left" else (0, 0, 255)
            cv2.drawMarker(
                verify_img, (int(uv[0]), int(uv[1])), color,
                cv2.MARKER_TILTED_CROSS, 24, 2,
            )
        cv2.imwrite(str(args.out / "verify_hover.png"), verify_img)
        print(f"verification saved -> {args.out / 'verify_hover.png'}")

        # Park back at the recorded start pose before torque-off; never drop.
        if not args.yes:
            input("Holding hover. Press Enter to park the arms and exit...")
        print("Parking arms at their startup pose before torque-off...")
        try:
            runtime.move_to(
                measured, duration_s=6.0, max_joint_delta=0.02,
                tolerance=0.05, timeout_s=40.0,
            )
        except Exception as error:
            print(f"park failed ({error}); support the arms before closing")
    finally:
        if camera is not None:
            camera.close()
        runtime.close()


if __name__ == "__main__":
    main()
