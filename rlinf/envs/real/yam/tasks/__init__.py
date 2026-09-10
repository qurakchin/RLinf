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

"""Gym registration and factory for RLinf-native YAM tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym

from rlinf.envs.real.registry import register_tasks
from rlinf.envs.real.yam.config import YamPicoConfig
from rlinf.envs.real.yam.dual_yam_joint_env import DualYamJointEnv
from rlinf.envs.real.yam.leader_intervention import DualYamLeaderIntervention


def create_dual_yam_joint_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    robot_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    """Build the base follower env and optional motorized-leader wrapper.

    The task id resolves through Gymnasium, which hands over the same five
    arguments ``rlinf.envs.real.registry`` generates for every other
    real-world task. YAM composes its intervention wrapper here rather than
    through ``build_stack`` because the VR and teaching-handle paths drive
    joints through YAM's own IK, not through a ``TeleopDevice``.
    """
    base_config = dict(override_cfg or {})
    leader_config_value = base_config.pop(
        "leader_intervention", base_config.pop("yam_leader", {})
    )
    leader_config = dict(leader_config_value or {})
    use_leaders = bool(
        base_config.pop(
            "use_yam_leader",
            leader_config.pop("enabled", False),
        )
    )
    use_pico = bool((env_cfg or {}).get("use_pico", False))
    if use_pico and use_leaders:
        raise ValueError(
            "YAM PICO and motorized leader intervention are mutually exclusive"
        )
    pico_config = (
        YamPicoConfig(**dict((env_cfg or {}).get("pico", {}))) if use_pico else None
    )
    if use_pico and not base_config.get("enforce_runtime_joint_limits", True):
        raise ValueError("YAM VR requires enforce_runtime_joint_limits=true")
    env = DualYamJointEnv(
        override_cfg=base_config,
        worker_info=worker_info,
        robot_info=robot_info,
        env_idx=env_idx,
    )

    if env_cfg is not None:
        main_image_key = env_cfg.get("main_image_key")
        frame_spaces = env.observation_space["frames"].spaces
        if main_image_key is not None and main_image_key not in frame_spaces:
            env.close()
            raise ValueError(
                f"YAM main_image_key {main_image_key!r} is not configured; "
                f"available cameras: {list(frame_spaces)}"
            )
    if use_leaders:
        return DualYamLeaderIntervention(env, leader_config)
    if use_pico:
        from rlinf.envs.real.yam.pico_intervention import DualYamPicoIntervention

        try:
            return DualYamPicoIntervention(env, pico_config)
        except Exception:
            env.close()
            raise
    return env


#: Mapping from Gymnasium IDs to the YAM entry points that build them.
TASKS: dict[str, Any] = {"DualYamJointEnv-v1": create_dual_yam_joint_env}

_ENTRY_POINTS = register_tasks(__name__, globals(), TASKS)

__all__ = ["TASKS", *_ENTRY_POINTS]
