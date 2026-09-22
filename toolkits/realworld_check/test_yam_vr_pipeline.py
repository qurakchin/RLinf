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

"""Run synthetic or live PICO through actual YAM IK and mock followers only.

This tool has no hardware execution option. --zmq-addr subscribes to live VR
but still uses mock CAN and mock cameras. Run as a module from the repo root.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from rlinf.envs.real.wrappers.teleop.composed import ComposedTeleop
from rlinf.envs.real.wrappers.teleop.facts import EnvFacts
from rlinf.envs.real.wrappers.teleop.intervention import TeleopIntervention
from rlinf.envs.real.wrappers.teleop.layout import action_spec
from rlinf.envs.real.yam.config import YamPicoConfig
from rlinf.envs.real.yam.dual_yam_joint_env import DualYamJointEnv
from rlinf.envs.real.yam.mock_backend import MockYamBackendFactory
from rlinf.envs.real.yam.pico_episode import YamPicoEpisode
from rlinf.robotics.parts.teleop.group import TeleopEntry, TeleopGroup
from rlinf.robotics.parts.teleop.yam_pico import YamPico
from rlinf.robotics.parts.transports.pico import PicoExpert
from rlinf.utils.logging import get_logger


class SyntheticExpert(PicoExpert):
    """Use the real mapping with deterministic JSON samples and no socket."""

    def start(self) -> None:
        """Suppress transport startup for the synthetic diagnostic."""

    def feed(self, step: int) -> None:
        """Update only this instance's cache, like its receive thread would."""
        controller = {
            "position": [0.01 * np.sin(max(0, step - 2) / 30), 1.0, -0.3],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "grip": float(step >= 2),
            "trigger": 0.0,
        }
        with self._lock:
            self._latest_data = {
                "headset_pose": [0.0, 1.6, 0.0, 0.0, 0.0, 0.0, 1.0],
                "left_controller": controller,
                "right_controller": controller,
                "buttons": {},
            }
            self._last_update_time = time.time()


def main() -> int:
    """Report real-model solve timing without opening robot devices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zmq-addr", help="Optional live publisher; output remains simulated"
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--frequency", type=float, default=30.0)
    args = parser.parse_args()
    if args.steps < 3 or not np.isfinite(args.frequency) or args.frequency <= 0:
        parser.error("steps must be >=3 and frequency finite and positive")
    logger = get_logger()
    config = YamPicoConfig(
        wait_for_record_button=False,
        **({"zmq_addr": args.zmq_addr} if args.zmq_addr else {}),
    )
    factory = MockYamBackendFactory()
    base = DualYamJointEnv(
        {
            "is_dummy": True,
            "step_frequency": args.frequency,
            "image_width": 8,
            "image_height": 8,
            "manual_episode_control_only": True,
        },
        backend_factory=factory,
    )
    experts = {}
    wrapper = None
    try:
        for side in ("left", "right"):
            cls = PicoExpert if args.zmq_addr else SyntheticExpert
            experts[side] = cls(**config.expert_kwargs(side))
        spec = action_spec(base)
        facts = EnvFacts.about(base, spec.layout, spec.kinds)
        device = YamPico(
            config,
            joint_step_limits=facts.joint_step_limits,
            joint_lower=facts.joint_limit_min,
            joint_upper=facts.joint_limit_max,
            experts=experts,
        )
        device.connect()
        group = TeleopGroup([TeleopEntry(device)], available=facts.kinds)
        wrapper = YamPicoEpisode(
            TeleopIntervention(
                base,
                ComposedTeleop(group, facts.layout, timeout=group.hold_window),
                mark_flag=base.TELEOP_MARK_FLAG,
            ),
            config,
        )
        wrapper.reset()
        # Directly seed only our explicitly created in-memory follower objects.
        for follower in factory.followers:
            follower.target = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5])
        times = []
        solves = []
        faults = {}
        active_steps = 0
        final = None
        for step in range(args.steps):
            if not args.zmq_addr:
                for expert in experts.values():
                    expert.feed(step)
            started = time.monotonic()
            _, _, _, _, info = wrapper.step(np.zeros(14))
            times.append(time.monotonic() - started)
            active_steps += int(info["pico_active"])
            for side in ("left", "right"):
                if f"{side}_ik_elapsed_s" in info:
                    solves.append(info[f"{side}_ik_elapsed_s"])
            if info["yam_pico_fault"]:
                key = info["yam_pico_fault"]
                faults[key] = faults.get(key, 0) + 1
            final = info["accepted_action"].tolist()

        def timing(values):
            return (
                dict(
                    zip(
                        ("p50", "p95", "p99", "max"),
                        (np.percentile(values, [50, 95, 99, 100]) * 1000).tolist(),
                        strict=True,
                    )
                )
                if values
                else {}
            )

        logger.info(
            "YAM VR offline report: %s",
            json.dumps(
                {
                    "hardware": "mock_only",
                    "steps": args.steps,
                    "active_steps": active_steps,
                    "ik_ms": timing(solves),
                    "step_including_pacing_ms": timing(times),
                    "steps_over_budget": int(
                        np.sum(np.array(times) > 1 / args.frequency)
                    ),
                    "faults": faults,
                    "last_accepted_action": final,
                }
            ),
        )
        return int(bool(faults) or active_steps == 0)
    finally:
        if wrapper is not None:
            wrapper.close()
        else:
            base.close()
            for expert in experts.values():
                expert.stop()


if __name__ == "__main__":
    raise SystemExit(main())
