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

"""Move both YAM followers to a named pose (reset / park / home primitive).

Defaults to the folded resting pose recorded on this station (arms folded at
the table edges, grippers open), which keeps the top camera view clear and is
safe to torque-off into. Pass --target-json with a 14-D list to use another
pose. With --hold, keep torque on and wait for Ctrl-C, then return to the
pose measured at startup before closing.

Usage:
    python toolkits/calibration/park_arms.py
    python toolkits/calibration/park_arms.py --target-json pose.json --hold
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from collect_calibration_data import _clear_motor_errors, _device  # noqa: E402

from rlinf.envs.real.yam.config import DualYamJointEnvConfig  # noqa: E402
from rlinf.envs.real.yam.control_runtime import YamControlRuntime  # noqa: E402
from rlinf.envs.real.yam.i2rt_backend import I2RTYamBackendFactory  # noqa: E402

# Folded resting pose measured on this station (calib_data/smoke sample 0),
# grippers overridden to open.
FOLDED_POSE = [
    -0.0032,
    0.0063,
    0.0055,
    -0.0685,
    0.0086,
    -0.0147,
    1.0,
    0.0216,
    0.0071,
    0.0078,
    -0.0795,
    -0.0261,
    0.0971,
    1.0,
]

_JOINT_LIMIT_MIN = [-2.61799, 0.0, 0.0, -1.69297, -1.5708, -2.0944]
_JOINT_LIMIT_MAX = [3.14159, 3.66519, 3.14159, 1.5708, 1.5708, 2.0944]


def _runtime() -> YamControlRuntime:
    cfg = DualYamJointEnvConfig(
        step_frequency=30.0,
        max_joint_delta=0.05,
        # See hover_over_pixels: the runtime slew clip starves PD torque; the
        # move_to interpolation already paces motion.
        enforce_runtime_joint_limits=False,
        joint_limit_min=[_JOINT_LIMIT_MIN] * 2,
        joint_limit_max=[_JOINT_LIMIT_MAX] * 2,
    )
    # Reuse the collector's device spec but with position control: park must
    # not float the arms, and the technician-tuned coulomb compensation helps
    # the elbow converge.
    left = _device("can_left")
    right = _device("can_right")
    for dev in (left, right):
        dev.zero_gravity_mode = False
        dev.use_coulomb_friction = True
    hardware = SimpleNamespace(left_follower=left, right_follower=right)
    return YamControlRuntime(cfg, hardware, I2RTYamBackendFactory())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-json",
        type=Path,
        default=None,
        help="JSON file with a 14-D target state.",
    )
    parser.add_argument(
        "--hold", action="store_true", help="Hold the pose until Ctrl-C, then re-park."
    )
    args = parser.parse_args()

    target = np.asarray(
        json.loads(args.target_json.read_text()) if args.target_json else FOLDED_POSE,
        dtype=np.float64,
    )
    if target.shape != (14,):
        raise ValueError("target must be 14-D")

    runtime = _runtime()
    try:
        _clear_motor_errors()
        runtime.connect_followers()
        runtime.hold()
        start = runtime.read_state().as_vector()
        print("moving to target pose ...")
        runtime.move_to(
            target,
            duration_s=6.0,
            max_joint_delta=0.02,
            tolerance=0.05,
            timeout_s=40.0,
        )
        print("target reached")
        if args.hold:
            try:
                input("Holding. Press Enter to return to the startup pose...")
            except KeyboardInterrupt:
                pass
            runtime.move_to(
                start,
                duration_s=6.0,
                max_joint_delta=0.02,
                tolerance=0.05,
                timeout_s=40.0,
            )
            print("returned to startup pose")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
