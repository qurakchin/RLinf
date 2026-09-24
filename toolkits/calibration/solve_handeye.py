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

"""Solve camera extrinsics from a collect_calibration_data.py capture.

Eye-in-hand (wrist cameras): pairs each detected board view with the measured
arm FK pose and runs cv2.calibrateHandEye, yielding T_grasp->cam. The static
top camera is then chained through the board: T_base->top = T_base->board x
(T_top->board)^-1, averaged over every sample where the top camera sees the
board.

Outputs <capture>/extrinsics.json with 4x4 matrices, quaternions, and
consistency residuals, plus <capture>/overlay/ images for visual inspection.

Usage:
    python toolkits/calibration/solve_handeye.py calib_data/run1 \
        --square-size 0.03 --pattern 11 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from board_detect import find_chessboard_custom  # noqa: E402

WRIST_CAMS = ("left", "right")  # left/right = wrist cameras of the same-side arm
ARM_SLICE = {"left": (0, 7), "right": (7, 14)}


def _rot_log(rmat: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle)."""
    angle = np.arccos(np.clip((np.trace(rmat) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-12:
        return np.zeros(3)
    axis = np.array(
        [rmat[2, 1] - rmat[1, 2], rmat[0, 2] - rmat[2, 0], rmat[1, 0] - rmat[0, 1]]
    ) / (2.0 * np.sin(angle))
    return axis * angle


def _left_qmat(q: np.ndarray) -> np.ndarray:
    """Matrix L(q) such that L(q) p == quaternion product q ⊗ p, [w,x,y,z]."""
    w, x, y, z = q
    return np.array(
        [
            [w, -x, -y, -z],
            [x, w, -z, y],
            [y, z, w, -x],
            [z, -y, x, w],
        ]
    )


def _right_qmat(q: np.ndarray) -> np.ndarray:
    """Matrix R(q) such that R(q) p == quaternion product p ⊗ q, [w,x,y,z]."""
    w, x, y, z = q
    return np.array(
        [
            [w, -x, -y, -z],
            [x, w, z, -y],
            [y, -z, w, x],
            [z, y, -x, w],
        ]
    )


def _rotmat_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _solve_hand_eye(
    r_g2b: list[np.ndarray],
    t_g2b: list[np.ndarray],
    r_t2c: list[np.ndarray],
    t_t2c: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """AX=XB solve for X = T_grasp->cam (eye-in-hand).

    A = inv(T_g2b(j)) @ T_g2b(i) and B = T_t2c(j) @ inv(T_t2c(i)), from
    T_g2b(i) @ X @ T_t2c(i) == T_g2b(j) @ X @ T_t2c(j), where the T_t2c inputs
    are the solvePnP board poses in the camera frame. Rotation is solved in
    quaternion form — q_a ⊗ q_x == q_x ⊗ q_b becomes a linear null-space
    problem — translation follows from (R_a - I) t_x = R_x t_b - t_a.
    Verified by ``--self-test``.
    """
    n = len(r_g2b)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    quat_rows = []
    for i, j in pairs:
        ra = r_g2b[j].T @ r_g2b[i]
        rb = r_t2c[j] @ r_t2c[i].T
        if np.linalg.norm(_rot_log(ra)) < 1e-6 or np.linalg.norm(_rot_log(rb)) < 1e-6:
            continue
        quat_rows.append(_left_qmat(_quat(ra)) - _right_qmat(_quat(rb)))
    if len(quat_rows) < 2:
        raise RuntimeError("not enough rotational diversity for hand-eye solve")
    _, _, vh = np.linalg.svd(np.concatenate(quat_rows))
    q_x = vh[-1]
    r_x = _rotmat_from_quat(q_x)

    # Translation: (R_a - I) t_x = R_x t_b - t_a
    rows, rhs = [], []
    for i, j in pairs:
        ra = r_g2b[j].T @ r_g2b[i]
        ta = r_g2b[j].T @ (t_g2b[i] - t_g2b[j])
        rb = r_t2c[j] @ r_t2c[i].T
        tb = t_t2c[j] - rb @ t_t2c[i]
        if np.linalg.norm(_rot_log(ra)) < 1e-6 or np.linalg.norm(_rot_log(rb)) < 1e-6:
            continue
        rows.append(ra - np.eye(3))
        rhs.append(r_x @ tb - ta)
    t_x = np.linalg.lstsq(np.concatenate(rows), np.concatenate(rhs), rcond=None)[0]
    return r_x, t_x


def _self_test() -> None:
    """Recover a known T_grasp->cam from synthetic poses."""
    rng = np.random.default_rng(7)
    truth = np.eye(4)
    truth[:3, :3] = cv2.Rodrigues(rng.normal(size=3))[0]
    truth[:3, 3] = rng.normal(scale=0.1, size=3)
    t_base_board = np.eye(4)
    t_base_board[:3, 3] = np.array([0.3, 0.1, 0.05])

    r_g2b, t_g2b, r_t2c, t_t2c = [], [], [], []
    for _ in range(15):
        t_g = np.eye(4)
        t_g[:3, :3] = cv2.Rodrigues(rng.normal(scale=0.5, size=3))[0]
        t_g[:3, 3] = rng.normal(scale=0.1, size=3)
        t_c = np.linalg.inv(t_g @ truth) @ t_base_board
        r_g2b.append(t_g[:3, :3])
        t_g2b.append(t_g[:3, 3])
        r_t2c.append(t_c[:3, :3])
        t_t2c.append(t_c[:3, 3])
    r_x, t_x = _solve_hand_eye(r_g2b, t_g2b, r_t2c, t_t2c)
    rot_err = np.linalg.norm(_rot_log(r_x @ truth[:3, :3].T))
    t_err = np.linalg.norm(t_x - truth[:3, 3])
    assert rot_err < 1e-8 and t_err < 1e-8, (rot_err, t_err)
    print("self-test passed (rot_err=%.2e, t_err=%.2e)" % (rot_err, t_err))


def _object_points(pattern: tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = pattern
    grid = np.zeros((rows * cols, 3), dtype=np.float64)
    for j in range(rows):
        for i in range(cols):
            grid[j * cols + i] = (i * square_size, j * square_size, 0.0)
    return grid


def _camera_matrix(intr: dict) -> np.ndarray:
    return np.array(
        [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]]
    )


def _board_in_cam(
    image: np.ndarray,
    pattern: tuple[int, int],
    obj_pts: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    """Detect the board; return (T_cam->board 4x4, mean reprojection error px)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = find_chessboard_custom(gray, pattern)
    if not found:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj_pts, corners, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    reproj = float(
        np.mean(np.linalg.norm(proj.reshape(-1, 2) - corners.reshape(-1, 2), axis=1))
    )
    rmat, _ = cv2.Rodrigues(rvec)
    t = np.eye(4)
    t[:3, :3] = rmat
    t[:3, 3] = tvec.reshape(3)
    return t, reproj


def _quat(matrix: np.ndarray) -> list[float]:
    """4x4 -> [w, x, y, z] quaternion."""
    r = matrix[:3, :3]
    tr = float(np.trace(r))
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return [
            0.25 * s,
            (r[2, 1] - r[1, 2]) / s,
            (r[0, 2] - r[2, 0]) / s,
            (r[1, 0] - r[0, 1]) / s,
        ]
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2
    q = [0.0] * 4
    q[0] = (r[k, j] - r[j, k]) / s
    q[i + 1] = 0.25 * s
    q[j + 1] = (r[j, i] + r[i, j]) / s
    q[k + 1] = (r[k, i] + r[i, k]) / s
    return [float(v) for v in q]


def _fk_all(kin, samples: list[dict], lo: int) -> list[np.ndarray]:
    """FK every sample's arm joints to T_base->tcp 4x4 poses."""
    out = []
    for sample in samples:
        state = np.asarray(sample["state"], dtype=np.float64)
        out.append(kin.fk(state[lo : lo + 6], float(state[lo + 6])))
    return out


def _mean_transform(transforms: list[np.ndarray]) -> np.ndarray:
    """Average 4x4 transforms: mean translation, quaternion-averaged rotation."""
    """Average 4x4 transforms: mean translation, quaternion-averaged rotation."""
    t = np.mean([m[:3, 3] for m in transforms], axis=0)
    # Markley quaternion averaging.
    quats = np.array([_quat(m) for m in transforms])
    quats[quats[:, 0] < 0] *= -1
    _, _, vh = np.linalg.svd(quats.T @ quats)
    w, x, y, z = vh[0]
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    out = np.eye(4)
    out[:3, :3] = r
    out[:3, 3] = t
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, nargs="?")
    parser.add_argument(
        "--pattern",
        type=int,
        nargs=2,
        default=(11, 8),
        help="Inner corner count (cols rows).",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=0.03,
        help="Measured square side length in meters.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Verify the hand-eye solver on synthetic data and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        _self_test()
        return
    if args.capture is None:
        raise SystemExit("capture directory is required")
    from rlinf.envs.real.yam.kinematics import YamKinematicsAdapter

    meta = json.loads((args.capture / "meta.json").read_text())
    samples = [
        json.loads(line)
        for line in (args.capture / "frames.jsonl").read_text().splitlines()
        if line.strip()
    ]
    pattern = tuple(args.pattern)
    obj_pts = _object_points(pattern, args.square_size)
    kin = YamKinematicsAdapter()

    overlay_dir = args.capture / "overlay"
    overlay_dir.mkdir(exist_ok=True)

    # Detect the board in every camera of every sample. Views whose solvePnP
    # reprojection error exceeds 1 px are mis-detections from the grid-fit
    # fallback and poison the least-squares solve, so drop them here.
    detections: dict[int, dict[str, np.ndarray]] = {}
    dropped = 0
    for sample in samples:
        idx = sample["sample"]
        per_cam = {}
        for cam in ("left", "right", "top"):
            path = args.capture / cam / f"sample_{idx:06d}.png"
            if not path.is_file():
                continue
            image = cv2.imread(str(path))
            intr = meta["cameras"][cam]
            found = _board_in_cam(
                image,
                pattern,
                obj_pts,
                _camera_matrix(intr),
                np.zeros(5),  # D405 color distortion is negligible at 640x480
            )
            if found is None:
                continue
            t, reproj = found
            if reproj > 1.0:
                dropped += 1
                continue
            per_cam[cam] = t
            vis = image.copy()
            cv2.drawFrameAxes(
                vis,
                _camera_matrix(intr),
                np.zeros(5),
                cv2.Rodrigues(t[:3, :3])[0],
                t[:3, 3],
                0.05,
            )
            cv2.imwrite(str(overlay_dir / f"{cam}_{idx:06d}.png"), vis)
        detections[idx] = per_cam
    counts = {
        cam: sum(1 for d in detections.values() if cam in d)
        for cam in ("left", "right", "top")
    }
    print(
        f"board detections per camera: {counts} / {len(samples)} samples "
        f"({dropped} dropped for reprojection error > 1 px)"
    )

    result: dict = {"pattern": pattern, "square_size_m": args.square_size}

    # --- Eye-in-hand for each wrist camera ---------------------------------
    for cam in WRIST_CAMS:
        lo, hi = ARM_SLICE[cam]
        usable = [
            sample for sample in samples if cam in detections.get(sample["sample"], {})
        ]
        if len(usable) < 5:
            print(f"{cam}: only {len(usable)} usable views; need >= 5, skipping")
            continue

        t_tcp_cam = None
        for _round in range(3):  # iterative outlier rejection on board pose
            r_g2b, t_g2b, r_t2c, t_t2c = [], [], [], []
            for sample in usable:
                state = np.asarray(sample["state"], dtype=np.float64)
                t_base_tcp = kin.fk(state[lo : lo + 6], float(state[lo + 6]))
                t_cam_board = detections[sample["sample"]][cam]
                r_g2b.append(t_base_tcp[:3, :3])
                t_g2b.append(t_base_tcp[:3, 3])
                r_t2c.append(t_cam_board[:3, :3])
                t_t2c.append(t_cam_board[:3, 3])
            r_c2g, t_c2g = _solve_hand_eye(r_g2b, t_g2b, r_t2c, t_t2c)
            t_tcp_cam = np.eye(4)
            t_tcp_cam[:3, :3] = r_c2g
            t_tcp_cam[:3, 3] = t_c2g.reshape(3)

            # Consistency: T_base->board should be constant across views.
            residuals = []
            for sample, t_base_tcp in zip(usable, _fk_all(kin, usable, lo)):
                residuals.append(
                    t_base_tcp @ t_tcp_cam @ detections[sample["sample"]][cam]
                )
            translations = np.array([m[:3, 3] for m in residuals])
            center = translations.mean(axis=0)
            dev = np.linalg.norm(translations - center, axis=1)
            if len(usable) <= 5 or dev.max() <= max(3 * dev.std(), 0.005):
                break
            keep = [
                sample
                for sample, d in zip(usable, dev)
                if d <= max(3 * dev.std(), 0.005)
            ]
            print(
                f"{cam}: rejecting {len(usable) - len(keep)} outlier view(s) "
                f"(max deviation {dev.max() * 1000:.0f} mm)"
            )
            usable = keep

        board_poses = []
        for sample, t_base_tcp in zip(usable, _fk_all(kin, usable, lo)):
            board_poses.append(
                t_base_tcp @ t_tcp_cam @ detections[sample["sample"]][cam]
            )
        translations = np.array([m[:3, 3] for m in board_poses])
        spread_mm = np.max(np.ptp(translations, axis=0)) * 1000.0
        std_mm = float(np.std(translations, axis=0).max()) * 1000.0
        result[f"{cam}_wrist"] = {
            "views_used": len(usable),
            "T_grasp_to_cam": t_tcp_cam.tolist(),
            "quat_wxyz": _quat(t_tcp_cam),
            "board_translation_spread_mm": float(spread_mm),
            "board_translation_std_mm": std_mm,
            "T_base_to_board_mean": _mean_transform(board_poses).tolist(),
        }
        print(
            f"{cam}: hand-eye solved from {len(usable)} views; "
            f"board pose consistency std {std_mm:.1f} mm, "
            f"spread {spread_mm:.1f} mm"
        )

    # --- Static top camera via the board ------------------------------------
    # Each arm chain yields T_base->top in ITS OWN base frame; comparing them
    # directly is meaningless. The meaningful cross-check converts the
    # right-chain result into the left base frame through the board:
    #   T_bL_bR = T_bL_board @ inv(T_bR_board), then compare
    #   T_bL_top_via_right = T_bL_bR @ T_bR_top  against  T_bL_top.
    top_base = {}
    for cam in WRIST_CAMS:
        key = f"{cam}_wrist"
        if key not in result:
            continue
        t_base_board = np.asarray(result[key]["T_base_to_board_mean"])
        per = []
        for sample in samples:
            idx = sample["sample"]
            if "top" in detections.get(idx, {}):
                per.append(t_base_board @ np.linalg.inv(detections[idx]["top"]))
        if per:
            top_base[cam] = _mean_transform(per)
    if top_base:
        result["top_camera"] = {
            "views_used": counts["top"],
            "note": "each via_<arm> entry is T_<arm>-base -> top camera",
            "T_base_to_cam": {
                f"via_{cam}_arm": m.tolist() for cam, m in top_base.items()
            },
        }
        for cam, m in top_base.items():
            result["top_camera"][f"quat_wxyz_via_{cam}_arm"] = _quat(m)
        if len(top_base) == 2:
            # Real consistency signal: with both camera and board static, the
            # per-sample T_base->top estimates must be constant; their spread
            # reflects top-camera solvePnP noise plus wrist-chain noise.
            for cam in WRIST_CAMS:
                key = f"{cam}_wrist"
                t_base_board = np.asarray(result[key]["T_base_to_board_mean"])
                per_t = np.array(
                    [
                        (t_base_board @ np.linalg.inv(detections[idx]["top"]))[:3, 3]
                        for idx in detections
                        if "top" in detections[idx]
                    ]
                )
                std_mm = float(np.std(per_t, axis=0).max()) * 1000.0
                spread_mm = float(np.max(np.ptp(per_t, axis=0))) * 1000.0
                result["top_camera"][f"per_sample_std_mm_via_{cam}"] = std_mm
                result["top_camera"][f"per_sample_spread_mm_via_{cam}"] = spread_mm
                print(
                    f"top camera per-sample consistency via {cam}: "
                    f"std {std_mm:.1f} mm, spread {spread_mm:.1f} mm "
                    f"({len(per_t)} views)"
                )

    (args.capture / "extrinsics.json").write_text(json.dumps(result, indent=2))
    print(f"saved -> {args.capture / 'extrinsics.json'}")


if __name__ == "__main__":
    main()
