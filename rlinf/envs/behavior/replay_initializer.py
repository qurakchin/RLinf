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

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf


@dataclass(frozen=True)
class ReplayEpisode:
    episode_index: int
    instance_id: int
    parquet_path: Path
    annotation_path: Path | None
    length: int | None = None


@dataclass(frozen=True)
class ReplayPlan:
    episode_index: int
    instance_id: int
    actions: np.ndarray
    replay_steps: int
    target_step: int


class BehaviorReplayInitializer:
    """Replay BEHAVIOR demonstration prefixes before policy rollout starts."""

    def __init__(self, cfg: DictConfig, seed_offset: int = 0):
        replay_cfg = OmegaConf.select(cfg, "replay_init")
        self.enabled = bool(
            replay_cfg is not None and OmegaConf.select(replay_cfg, "enabled", default=False)
        )
        if not self.enabled:
            return

        self.dataset_root = Path(
            OmegaConf.select(replay_cfg, "dataset_root", default="")
        ).expanduser()
        if not self.dataset_root.is_dir():
            raise ValueError(
                f"env.replay_init.dataset_root must be an existing directory, got {self.dataset_root}"
            )

        self.task_id = int(OmegaConf.select(replay_cfg, "task_id", default=0))
        self.task_dir = f"task-{self.task_id:04d}"
        self.action_column = str(
            OmegaConf.select(replay_cfg, "action_column", default="action")
        )
        self.stage_index = OmegaConf.select(replay_cfg, "stage_index", default=None)
        if self.stage_index is not None:
            self.stage_index = int(self.stage_index)
        self.stage_boundary = str(
            OmegaConf.select(replay_cfg, "stage_boundary", default="start")
        ).lower()
        if self.stage_boundary not in ("start", "end"):
            raise ValueError("env.replay_init.stage_boundary must be 'start' or 'end'.")
        self.target_step = OmegaConf.select(replay_cfg, "target_step", default=None)
        if self.target_step is not None:
            self.target_step = int(self.target_step)

        self.replay_ratio = float(
            OmegaConf.select(replay_cfg, "replay_ratio", default=1.0)
        )
        if self.replay_ratio < 0:
            raise ValueError("env.replay_init.replay_ratio must be non-negative.")
        self.min_replay_steps = int(
            OmegaConf.select(replay_cfg, "min_replay_steps", default=0)
        )
        self.max_replay_steps = OmegaConf.select(
            replay_cfg, "max_replay_steps", default=None
        )
        if self.max_replay_steps is not None:
            self.max_replay_steps = int(self.max_replay_steps)
        self.jitter_steps = int(OmegaConf.select(replay_cfg, "jitter_steps", default=0))

        self.noise_std = float(OmegaConf.select(replay_cfg, "noise_std", default=0.0))
        self.noise_clip = OmegaConf.select(replay_cfg, "noise_clip", default=None)
        if self.noise_clip is not None:
            self.noise_clip = float(self.noise_clip)
        self.action_clip = OmegaConf.select(replay_cfg, "action_clip", default=None)
        if isinstance(self.action_clip, ListConfig):
            self.action_clip = OmegaConf.to_container(self.action_clip)
        if self.action_clip is not None:
            if len(self.action_clip) != 2:
                raise ValueError("env.replay_init.action_clip must be [low, high].")
            self.action_clip = (float(self.action_clip[0]), float(self.action_clip[1]))

        try:
            seed_value = OmegaConf.select(replay_cfg, "seed", default=None)
        except Exception:
            seed_value = None
        if seed_value is None:
            seed_value = cfg.get("seed", 0)
        seed = int(seed_value) + int(seed_offset)
        self._py_rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._action_cache: dict[int, np.ndarray] = {}
        self.allowed_instance_ids = self._resolve_allowed_instance_ids(cfg)

        requested_episode_ids = OmegaConf.select(
            replay_cfg, "episode_ids", default=None
        )
        if isinstance(requested_episode_ids, ListConfig):
            requested_episode_ids = OmegaConf.to_container(requested_episode_ids)
        if requested_episode_ids is not None:
            requested_episode_ids = [int(x) for x in requested_episode_ids]

        self.episodes = self._discover_episodes(requested_episode_ids)
        if not self.episodes:
            raise ValueError(
                f"No replay episodes found under {self.dataset_root / 'data' / self.task_dir}."
            )

    def sample_plans(self, num_envs: int) -> list[ReplayPlan]:
        episodes = [self._py_rng.choice(self.episodes) for _ in range(num_envs)]
        return [self._build_plan(episode) for episode in episodes]

    def _discover_episodes(
        self, requested_episode_ids: list[int] | None
    ) -> list[ReplayEpisode]:
        data_dir = self.dataset_root / "data" / self.task_dir
        annotation_dir = self.dataset_root / "annotations" / self.task_dir
        meta_by_episode = self._load_episode_lengths()

        episodes = []
        requested = set(requested_episode_ids) if requested_episode_ids else None
        for parquet_path in sorted(data_dir.glob("episode_*.parquet")):
            episode_index = self._parse_episode_index(parquet_path.stem)
            if requested is not None and episode_index not in requested:
                continue
            instance_id = self._episode_index_to_instance_id(episode_index)
            if (
                self.allowed_instance_ids is not None
                and instance_id not in self.allowed_instance_ids
            ):
                continue
            annotation_path = annotation_dir / f"episode_{episode_index:08d}.json"
            episodes.append(
                ReplayEpisode(
                    episode_index=episode_index,
                    instance_id=instance_id,
                    parquet_path=parquet_path,
                    annotation_path=annotation_path if annotation_path.is_file() else None,
                    length=meta_by_episode.get(episode_index),
                )
            )
        return episodes

    @staticmethod
    def _resolve_allowed_instance_ids(cfg: DictConfig) -> set[int] | None:
        instance_ids = OmegaConf.select(
            cfg, "omni_config.task.activity_instance_id", default=None
        )
        if isinstance(instance_ids, ListConfig):
            instance_ids = OmegaConf.to_container(instance_ids, resolve=True)
        if isinstance(instance_ids, int):
            return {int(instance_ids)}
        if isinstance(instance_ids, (list, tuple)):
            allowed = set()
            for item in instance_ids:
                if isinstance(item, int):
                    allowed.add(int(item))
            return allowed or None
        return None

    def _load_episode_lengths(self) -> dict[int, int]:
        path = self.dataset_root / "meta" / "episodes.jsonl"
        if not path.is_file():
            return {}
        lengths = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                episode_index = int(item["episode_index"])
                if episode_index // 10000 == self.task_id:
                    lengths[episode_index] = int(item.get("length", 0))
        return lengths

    def _build_plan(self, episode: ReplayEpisode) -> ReplayPlan:
        actions = self._load_actions(episode)
        target_step = self._resolve_target_step(episode, len(actions))
        replay_steps = int(round(target_step * self.replay_ratio))
        if self.jitter_steps > 0:
            replay_steps += self._py_rng.randint(-self.jitter_steps, self.jitter_steps)
        replay_steps = max(self.min_replay_steps, replay_steps)
        if self.max_replay_steps is not None:
            replay_steps = min(self.max_replay_steps, replay_steps)
        replay_steps = min(max(replay_steps, 0), len(actions))

        replay_actions = actions[:replay_steps].copy()
        if replay_actions.size and self.noise_std > 0:
            noise = self._np_rng.normal(
                loc=0.0, scale=self.noise_std, size=replay_actions.shape
            ).astype(replay_actions.dtype, copy=False)
            if self.noise_clip is not None:
                noise = np.clip(noise, -self.noise_clip, self.noise_clip)
            replay_actions += noise
        if replay_actions.size and self.action_clip is not None:
            replay_actions = np.clip(
                replay_actions, self.action_clip[0], self.action_clip[1]
            )

        return ReplayPlan(
            episode_index=episode.episode_index,
            instance_id=episode.instance_id,
            actions=replay_actions,
            replay_steps=replay_steps,
            target_step=target_step,
        )

    def _load_actions(self, episode: ReplayEpisode) -> np.ndarray:
        cached = self._action_cache.get(episode.episode_index)
        if cached is not None:
            return cached

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "env.replay_init requires pyarrow to read BEHAVIOR parquet files."
            ) from exc

        table = pq.read_table(episode.parquet_path, columns=[self.action_column])
        values = table[self.action_column].to_pylist()
        actions = np.asarray(values, dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(
                f"Replay action column {self.action_column!r} in {episode.parquet_path} "
                f"must be 2-D after loading, got shape {actions.shape}."
            )
        self._action_cache[episode.episode_index] = actions
        return actions

    def _resolve_target_step(self, episode: ReplayEpisode, action_count: int) -> int:
        if self.target_step is not None:
            return min(max(self.target_step, 0), action_count)
        if self.stage_index is not None:
            return min(max(self._load_stage_step(episode), 0), action_count)
        if episode.length is not None and episode.length > 0:
            return min(episode.length, action_count)
        return action_count

    def _load_stage_step(self, episode: ReplayEpisode) -> int:
        if episode.annotation_path is None:
            raise ValueError(
                f"stage_index was set but no annotation file exists for episode {episode.episode_index}."
            )
        with episode.annotation_path.open("r", encoding="utf-8") as f:
            annotation = json.load(f)
        skills = annotation.get("skill_annotation", [])
        if not skills:
            raise ValueError(f"No skill_annotation entries in {episode.annotation_path}.")
        idx = self.stage_index
        if idx < 0:
            idx = len(skills) + idx
        if idx < 0 or idx >= len(skills):
            raise ValueError(
                f"stage_index={self.stage_index} is out of range for "
                f"{episode.annotation_path} with {len(skills)} stages."
            )
        duration = skills[idx].get("frame_duration")
        if not isinstance(duration, list) or len(duration) != 2:
            raise ValueError(
                f"Invalid frame_duration for stage {self.stage_index} in {episode.annotation_path}."
            )
        return int(duration[0] if self.stage_boundary == "start" else duration[1])

    def _episode_index_to_instance_id(self, episode_index: int) -> int:
        task_offset = self.task_id * 10000
        within_task_index = episode_index - task_offset
        if within_task_index <= 0:
            raise ValueError(
                f"episode_index={episode_index} does not belong to task_id={self.task_id}."
            )
        if within_task_index % 10 == 0:
            return within_task_index // 10
        return within_task_index

    @staticmethod
    def _parse_episode_index(stem: str) -> int:
        prefix = "episode_"
        if not stem.startswith(prefix):
            raise ValueError(f"Invalid BEHAVIOR episode filename stem: {stem}")
        return int(stem[len(prefix) :])


def maybe_make_replay_initializer(
    cfg: DictConfig, seed_offset: int = 0
) -> BehaviorReplayInitializer | None:
    initializer = BehaviorReplayInitializer(cfg, seed_offset=seed_offset)
    return initializer if initializer.enabled else None


def replay_plans_to_infos(plans: list[ReplayPlan]) -> list[dict[str, Any]]:
    return [
        {
            "replay_episode_index": plan.episode_index,
            "replay_instance_id": plan.instance_id,
            "replay_steps": plan.replay_steps,
            "replay_target_step": plan.target_step,
        }
        for plan in plans
    ]
