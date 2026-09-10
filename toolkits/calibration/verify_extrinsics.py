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

"""End-to-end extrinsics verification: project each arm's FK TCP into the
top camera image and check the marker lands on the physical gripper.

Uses independent data (no board detection involved): the projected point
combines measured joints -> FK -> T_base->top from extrinsics.json.

Usage:
    python toolkits/calibration/verify_extrinsics.py calib_data/run1
Writes calib_data/run1/verify/verify_XXXXXX.png and prints pixel offsets of
the projected points from the image center for reference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ARM_SLICE = {"left": (0, 7), "right": (7, 14)}


def main() -> None:
    capture = Path(sys.argv[1] if len(sys.argv) > 1 else "calib_data/run1")
    n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    from rlinf.envs.real.yam.kinematics import YamKinematicsAdapter

    meta = json.loads((capture / "meta.json").read_text())
    extr = json.loads((capture / "extrinsics.json").read_text())
    samples = [
        json.loads(line)
        for line in (capture / "frames.jsonl").read_text().splitlines()
        if line.strip()
    ]
    intr = meta["cameras"]["top"]
    K = np.array(
        [[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]]
    )
    t_base_top = {
        cam: np.asarray(extr["top_camera"]["T_base_to_cam"][f"via_{cam}_arm"])
        for cam in ("left", "right")
    }
    # extrinsics.json stores T_base->top (maps camera coords into the base
    # frame); projection needs the inverse.
    t_top_base = {cam: np.linalg.inv(m) for cam, m in t_base_top.items()}
    kin = YamKinematicsAdapter()

    out_dir = capture / "verify"
    out_dir.mkdir(exist_ok=True)
    picks = [
        samples[i]
        for i in np.linspace(0, len(samples) - 1, min(n_show, len(samples))).astype(int)
    ]
    for sample in picks:
        idx = sample["sample"]
        image = cv2.imread(str(capture / "top" / f"sample_{idx:06d}.png"))
        if image is None:
            continue
        state = np.asarray(sample["state"], dtype=np.float64)
        for cam, color in (("left", (0, 255, 0)), ("right", (0, 0, 255))):
            lo, _ = ARM_SLICE[cam]
            t_base_tcp = kin.fk(state[lo : lo + 6], float(state[lo + 6]))
            p_cam = t_top_base[cam] @ np.append(t_base_tcp[:3, 3], 1.0)
            uv = K @ p_cam[:3] / p_cam[2]
            cv2.drawMarker(
                image, (int(uv[0]), int(uv[1])), color,
                cv2.MARKER_TILTED_CROSS, 24, 2,
            )
            cv2.putText(image, cam, (int(uv[0]) + 8, int(uv[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(str(out_dir / f"verify_{idx:06d}.png"), image)
        print(f"verify_{idx:06d}.png saved")


if __name__ == "__main__":
    main()
