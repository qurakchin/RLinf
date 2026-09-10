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

"""Tests for CollectEpisode's streaming LeRobot write path."""

from __future__ import annotations

import json
import pickle
import threading
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

lerobot = pytest.importorskip("lerobot", reason="streaming tests need lerobot")

import rlinf.envs.wrappers.collect_episode as collect_episode_module  # noqa: E402
from rlinf.envs.wrappers import CollectEpisode  # noqa: E402

_IMAGE_SHAPE = (64, 64, 3)
_STATE_DIM = 14


class _ScriptedEnv(gym.Env):
    """Single-env fake whose step outcomes come from a script."""

    def __init__(self, script):
        super().__init__()
        self.closed = False
        self.action_space = spaces.Box(-1.0, 1.0, shape=(_STATE_DIM,))
        self.observation_space = spaces.Dict(
            {
                "main_images": spaces.Box(
                    0, 255, shape=(1, *_IMAGE_SHAPE), dtype=np.uint8
                ),
                "states": spaces.Box(-10.0, 10.0, shape=(1, _STATE_DIM)),
            }
        )
        self._script = list(script)
        self._step_count = 0

    def _obs(self):
        image = np.full((1, *_IMAGE_SHAPE), self._step_count % 255, dtype=np.uint8)
        states = np.full((1, _STATE_DIM), float(self._step_count), dtype=np.float32)
        return {
            "main_images": image,
            "states": states,
            "task_descriptions": ["stream_task"],
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        return self._obs(), {}

    def step(self, action):
        self._step_count += 1
        terminated, truncated, info = (
            self._script.pop(0)
            if self._script
            else (np.array([False]), np.array([False]), {})
        )
        reward = np.array([1.0 if terminated.any() else 0.0], dtype=np.float32)
        return self._obs(), reward, terminated, truncated, info

    def close(self):
        self.closed = True


class _ThreeViewScriptedEnv(_ScriptedEnv):
    """Single-env fake with one top and two extra-view RGB cameras."""

    def __init__(self, script):
        super().__init__(script)
        self.observation_space = spaces.Dict(
            {
                "main_images": spaces.Box(
                    0, 255, shape=(1, *_IMAGE_SHAPE), dtype=np.uint8
                ),
                "extra_view_images": spaces.Box(
                    0, 255, shape=(1, 2, *_IMAGE_SHAPE), dtype=np.uint8
                ),
                "states": spaces.Box(-10.0, 10.0, shape=(1, _STATE_DIM)),
            }
        )

    def _obs(self):
        top = np.full((1, *_IMAGE_SHAPE), self._step_count % 255, dtype=np.uint8)
        left = np.full(_IMAGE_SHAPE, (self._step_count * 3 + 11) % 255, dtype=np.uint8)
        right = np.full(_IMAGE_SHAPE, (self._step_count * 5 + 23) % 255, dtype=np.uint8)
        states = np.full((1, _STATE_DIM), float(self._step_count), dtype=np.float32)
        return {
            "main_images": top,
            "extra_view_images": np.stack([left, right], axis=0)[None, ...],
            "states": states,
            "task_descriptions": ["stream_task"],
        }


def _collect(tmp_path: Path, script) -> Path:
    save_dir = tmp_path / "collected"
    env = CollectEpisode(
        _ScriptedEnv(script),
        save_dir=str(save_dir),
        num_envs=1,
        export_format="lerobot",
        robot_type="dual_yam",
        fps=30,
        streaming=True,
    )
    env.reset()
    action = np.zeros((1, _STATE_DIM), dtype=np.float32)
    while env.unwrapped._script:
        env.step(action)
    env.close()
    return save_dir


def _episode_frame_table(shard: Path, episode_index: int):
    pq = pytest.importorskip("pyarrow.parquet")
    table = pq.read_table(
        shard / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    )
    return table


def _no_published_episodes(shard: Path) -> bool:
    return not list(shard.glob("data/**/*.parquet"))


def _sidecar_rows(shard: Path) -> list[dict]:
    sidecar = shard / "stream_frames.jsonl"
    if not sidecar.exists() or not sidecar.read_text().strip():
        return []
    return [json.loads(line) for line in sidecar.read_text().splitlines()]


def _drain_frame_executor(env: CollectEpisode) -> None:
    assert env._executor is not None
    env._executor.submit(lambda: None).result(timeout=2)


def _mp4_frame_count(path: Path, av_module) -> int:
    with av_module.open(str(path)) as container:
        stream = container.streams.video[0]
        return sum(1 for _ in container.decode(stream))


def test_streaming_mp4_export_uses_review_dir_outside_lerobot_root(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("PIL.Image")
    pytest.importorskip("pyarrow.parquet")
    av_module = pytest.importorskip("av")

    save_dir = tmp_path / "collected"
    script = [
        (np.array([False]), np.array([False]), {}),
        (np.array([True]), np.array([False]), {"success": np.array([True])}),
        (np.array([False]), np.array([False]), {}),
        (np.array([True]), np.array([False]), {"success": np.array([True])}),
    ]
    env = CollectEpisode(
        _ThreeViewScriptedEnv(script),
        save_dir=str(save_dir),
        num_envs=1,
        export_format="lerobot",
        robot_type="dual_yam",
        fps=30,
        streaming=True,
        export_mp4=True,
    )
    action = np.zeros((1, _STATE_DIM), dtype=np.float32)
    primary_error = None

    try:
        env.reset()
        env.step(action)
        env.step(action)
        env._wait_futures()
        env._wait_save_futures()
        assert env._video_executor is not None
        env._video_executor.submit(lambda: None).result(timeout=30)

        assert len(list(save_dir.rglob("*.mp4"))) == 3

        env.reset()
        env.step(action)
        env.step(action)
        env._wait_futures()
        env._wait_save_futures()
        assert env._video_executor is not None
        env._video_executor.submit(lambda: None).result(timeout=30)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            env.close()
        except BaseException:
            if primary_error is None:
                raise

    for shard_id in range(2):
        shard = save_dir / "rank_0" / f"id_{shard_id}"
        info = json.loads((shard / "meta" / "info.json").read_text())
        assert info["total_episodes"] == 1
        assert _episode_frame_table(shard, 0).num_rows == 2
        assert not list(shard.rglob("*.mp4"))

        episode_dir = (
            save_dir / "rank_0" / "review_videos" / f"id_{shard_id}" / "episode_000000"
        )
        for view_name in ("top", "left", "right"):
            mp4_path = episode_dir / f"{view_name}.mp4"
            assert mp4_path.is_file()
            assert _mp4_frame_count(mp4_path, av_module) == 2
    assert len(list((save_dir / "rank_0" / "review_videos").rglob("*.mp4"))) == 6


def test_streaming_writes_successful_episode(tmp_path):
    script = [
        (np.array([False]), np.array([False]), {}),
        (np.array([True]), np.array([False]), {"success": np.array([True])}),
    ]
    save_dir = _collect(tmp_path, script)

    shard = save_dir / "rank_0" / "id_0"
    info = json.loads((shard / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1

    table = _episode_frame_table(shard, 0)
    assert table.num_rows == 2
    assert all(bool(v) for v in table["is_success"].to_pylist()), (
        "episode-level success must be stamped on every frame"
    )
    done = [bool(v) for v in table["done"].to_pylist()]
    assert done == [False, True]
    tasks = (shard / "meta" / "tasks.jsonl").read_text()
    assert "stream_task" in tasks

    # The per-frame state/actions sidecar is written per frame, so it survives
    # even if a later parquet write is interrupted.
    sidecar = (shard / "stream_frames.jsonl").read_text().strip().splitlines()
    assert len(sidecar) == 2
    first = json.loads(sidecar[0])
    assert first["episode"] == 0 and first["frame"] == 0
    assert len(first["state"]) == _STATE_DIM
    assert len(first["actions"]) == _STATE_DIM


def test_streaming_writes_aborted_episode_anyway(tmp_path):
    script = [
        (np.array([False]), np.array([False]), {}),
        # record_reset discards the in-progress recording; write-all keeps it.
        (np.array([False]), np.array([False]), {"record_reset": np.array([True])}),
        (np.array([True]), np.array([False]), {"success": np.array([True])}),
    ]
    save_dir = _collect(tmp_path, script)

    aborted_shard = save_dir / "rank_0" / "id_0"
    aborted_info = json.loads((aborted_shard / "meta" / "info.json").read_text())
    assert aborted_info["total_episodes"] == 1

    aborted = _episode_frame_table(aborted_shard, 0)
    assert aborted.num_rows == 1
    assert not any(bool(v) for v in aborted["is_success"].to_pylist())

    succeeded_shard = save_dir / "rank_0" / "id_1"
    succeeded_info = json.loads((succeeded_shard / "meta" / "info.json").read_text())
    assert succeeded_info["total_episodes"] == 1

    succeeded = _episode_frame_table(succeeded_shard, 0)
    assert succeeded.num_rows == 1
    assert all(bool(v) for v in succeeded["is_success"].to_pylist())


def test_streaming_review_discard_then_keep_reuses_episode_zero_and_exports_only_keep(
    tmp_path, monkeypatch
):
    export_calls = []

    def record_export(parquet_path, output_dir):
        export_calls.append((parquet_path, output_dir))

    monkeypatch.setattr(
        CollectEpisode, "_export_episode_mp4", staticmethod(record_export)
    )
    script = [
        (np.array([False]), np.array([False]), {}),
        (np.array([False]), np.array([False]), {}),
        (
            np.array([False]),
            np.array([False]),
            {
                "episode_review_pending": np.array([True]),
                "pre_record": np.array([False]),
            },
        ),
        (
            np.array([False]),
            np.array([False]),
            {
                "episode_review_pending": np.array([True]),
                "pre_record": np.array([True]),
            },
        ),
        (
            np.array([True]),
            np.array([False]),
            {
                "episode_review_pending": np.array([False]),
                "episode_discarded": np.array([True]),
                "success": np.array([False]),
                "manual_done": np.array([False]),
                "pre_record": np.array([True]),
            },
        ),
        (np.array([False]), np.array([False]), {}),
        (
            np.array([False]),
            np.array([False]),
            {
                "episode_review_pending": np.array([True]),
                "pre_record": np.array([False]),
            },
        ),
        (
            np.array([True]),
            np.array([False]),
            {
                "episode_review_pending": np.array([False]),
                "episode_discarded": np.array([False]),
                "success": np.array([True]),
                "manual_done": np.array([True]),
                "pre_record": np.array([True]),
            },
        ),
    ]
    save_dir = tmp_path / "collected"
    env = CollectEpisode(
        _ScriptedEnv(script),
        save_dir=str(save_dir),
        num_envs=1,
        export_format="lerobot",
        robot_type="dual_yam",
        fps=30,
        streaming=True,
        export_mp4=True,
    )
    action = np.zeros((1, _STATE_DIM), dtype=np.float32)

    try:
        env.reset()
        for _ in range(3):
            env.step(action)
        env._wait_futures()
        shard = save_dir / "rank_0" / "id_0"
        assert _no_published_episodes(shard)
        assert [row["frame"] for row in _sidecar_rows(shard)] == [0, 1, 2]

        env.step(action)
        env.step(action)
        env.reset()
        env.step(action)
        env.step(action)
        env.step(action)
    finally:
        env.close()

    info = json.loads((shard / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1
    assert not list(shard.glob("images/**/*.png"))
    assert len(export_calls) == 1
    assert export_calls[0][0].name == "episode_000000.parquet"

    table = _episode_frame_table(shard, 0)
    assert table.num_rows == 2
    assert all(bool(v) for v in table["is_success"].to_pylist())
    assert [row["frame"] for row in _sidecar_rows(shard)] == [0, 1]


@pytest.mark.parametrize("finish_with", ["reset", "close"])
def test_streaming_review_pending_is_discarded_on_reset_or_close(tmp_path, finish_with):
    script = [
        (np.array([False]), np.array([False]), {}),
        (
            np.array([False]),
            np.array([False]),
            {
                "episode_review_pending": np.array([True]),
                "pre_record": np.array([False]),
            },
        ),
    ]
    save_dir = tmp_path / "collected"
    env = CollectEpisode(
        _ScriptedEnv(script),
        save_dir=str(save_dir),
        num_envs=1,
        export_format="lerobot",
        robot_type="dual_yam",
        fps=30,
        streaming=True,
    )
    action = np.zeros((1, _STATE_DIM), dtype=np.float32)

    env.reset()
    env.step(action)
    env.step(action)
    if finish_with == "reset":
        env.reset()
    env.close()

    shard = save_dir / "rank_0" / "id_0"
    assert _no_published_episodes(shard)
    assert _sidecar_rows(shard) == []
    assert not list(shard.glob("images/**/*.png"))
    assert not list(shard.glob("images/**/episode_*"))
    info_path = shard / "meta" / "info.json"
    if info_path.exists():
        assert json.loads(info_path.read_text())["total_episodes"] == 0


def test_streaming_queue_overflow_invalidates_episode_without_blocking(
    tmp_path, monkeypatch
):
    release_writer = threading.Event()
    writer_entered = threading.Event()
    submitted_frames = []
    original_add_frame = CollectEpisode._stream_add_frame

    def blocked_add_frame(self, frame):
        submitted_frames.append(frame)
        writer_entered.set()
        release_writer.wait(timeout=1.0)
        return original_add_frame(self, frame)

    monkeypatch.setattr(collect_episode_module, "_MAX_PENDING_FUTURES", 2)
    monkeypatch.setattr(CollectEpisode, "_stream_add_frame", blocked_add_frame)

    save_dir = tmp_path / "collected"
    env = CollectEpisode(
        _ScriptedEnv(
            [
                (np.array([False]), np.array([False]), {}),
                (np.array([False]), np.array([False]), {}),
                (np.array([False]), np.array([False]), {}),
                (np.array([False]), np.array([False]), {}),
                (np.array([True]), np.array([False]), {"success": np.array([True])}),
                (np.array([False]), np.array([False]), {}),
                (np.array([True]), np.array([False]), {"success": np.array([True])}),
            ]
        ),
        save_dir=str(save_dir),
        num_envs=1,
        export_format="lerobot",
        robot_type="dual_yam",
        fps=30,
        streaming=True,
    )
    action = np.zeros((1, _STATE_DIM), dtype=np.float32)

    try:
        env.reset()
        env.step(action)
        assert writer_entered.wait(timeout=1.0), "writer thread did not block"
        env.step(action)

        started = time.perf_counter()
        _, _, _, _, overflow_info = env.step(action)
        overflow_elapsed = time.perf_counter() - started

        assert overflow_elapsed < 0.2
        assert bool(np.asarray(overflow_info["recording_invalid"]).any())
        assert len(submitted_frames) <= 2

        _, _, _, _, still_invalid_info = env.step(action)
        assert bool(np.asarray(still_invalid_info["recording_invalid"]).any())

        _, _, terminated, truncated, done_info = env.step(action)
        assert bool(np.asarray(terminated).any()) and not bool(
            np.asarray(truncated).any()
        )
        assert bool(np.asarray(done_info["recording_invalid"]).any())

        release_writer.set()
        env._wait_futures()
        env.reset()
        _, _, _, _, fresh_info = env.step(action)
        assert not bool(np.asarray(fresh_info["recording_invalid"]).any())
        _, _, _, _, success_info = env.step(action)
        assert not bool(np.asarray(success_info["recording_invalid"]).any())
    finally:
        release_writer.set()
        env.close()

    shard = save_dir / "rank_0" / "id_0"
    info = json.loads((shard / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1

    errors_path = shard / "recording_errors.jsonl"
    errors = [json.loads(line) for line in errors_path.read_text().splitlines()]
    assert errors
    assert errors[0]["episode_index"] == 0
    assert errors[0]["reason"] == "recording_queue_overflow"
    assert 0 < errors[0]["saved_frames"] <= 2

    archive = Path(errors[0]["archive"])
    assert archive.is_dir()
    assert archive.parent == shard / "invalid_episodes"
    with (archive / "frames.pkl").open("rb") as frames_file:
        invalid_prefix = pickle.load(frames_file)
    assert invalid_prefix["size"] == errors[0]["saved_frames"]
    assert 0 < invalid_prefix["size"] <= 2
    assert all(str(path).startswith(str(archive)) for path in invalid_prefix["image"])
    assert all(Path(path).is_file() for path in invalid_prefix["image"])

    sidecar = [
        json.loads(line)
        for line in (shard / "stream_frames.jsonl").read_text().splitlines()
    ]
    assert len(sidecar) == 2
    assert [row["frame"] for row in sidecar] == [0, 1]

    recovered_episode = _episode_frame_table(shard, 0)
    assert recovered_episode.num_rows == 2
    assert all(bool(v) for v in recovered_episode["is_success"].to_pylist())


def test_streaming_blocked_save_does_not_fill_frame_queue(tmp_path, monkeypatch):
    from rlinf.data.storage.lerobot.writer import LeRobotDatasetWriter

    release_save = threading.Event()
    first_save_started = threading.Event()
    save_calls = 0
    original_save_episode = LeRobotDatasetWriter.save_episode

    def blocked_first_save(self, is_success=None):
        nonlocal save_calls
        save_calls += 1
        if save_calls == 1:
            first_save_started.set()
            release_save.wait(timeout=5.0)
        return original_save_episode(self, is_success=is_success)

    monkeypatch.setattr(collect_episode_module, "_MAX_PENDING_FUTURES", 2)
    monkeypatch.setattr(LeRobotDatasetWriter, "save_episode", blocked_first_save)

    save_dir = tmp_path / "collected"
    env = CollectEpisode(
        _ScriptedEnv(
            [
                (np.array([False]), np.array([False]), {}),
                (np.array([True]), np.array([False]), {"success": np.array([True])}),
                (np.array([False]), np.array([False]), {}),
                (np.array([False]), np.array([False]), {}),
                (np.array([False]), np.array([False]), {}),
                (np.array([False]), np.array([False]), {}),
                (np.array([True]), np.array([False]), {"success": np.array([True])}),
            ]
        ),
        save_dir=str(save_dir),
        num_envs=1,
        export_format="lerobot",
        robot_type="dual_yam",
        fps=30,
        streaming=True,
    )
    action = np.zeros((1, _STATE_DIM), dtype=np.float32)

    try:
        env.reset()
        env.step(action)
        env.step(action)
        env._wait_futures()
        assert first_save_started.wait(timeout=1.0)

        env.reset()
        for _ in range(4):
            _, _, _, _, info = env.step(action)
            _drain_frame_executor(env)
            assert not bool(np.asarray(info["recording_invalid"]).any())
        _, _, terminated, truncated, info = env.step(action)
        _drain_frame_executor(env)
        assert bool(np.asarray(terminated).any()) and not bool(
            np.asarray(truncated).any()
        )
        assert not bool(np.asarray(info["recording_invalid"]).any())
        env._wait_futures()

        second_shard = save_dir / "rank_0" / "id_1"
        assert [row["frame"] for row in _sidecar_rows(second_shard)] == [
            0,
            1,
            2,
            3,
            4,
        ]
    finally:
        release_save.set()
        env.close()

    first_shard = save_dir / "rank_0" / "id_0"
    second_shard = save_dir / "rank_0" / "id_1"
    assert (
        json.loads((first_shard / "meta" / "info.json").read_text())["total_episodes"]
        == 1
    )
    assert (
        json.loads((second_shard / "meta" / "info.json").read_text())["total_episodes"]
        == 1
    )
    assert _episode_frame_table(first_shard, 0).num_rows == 2
    assert _episode_frame_table(second_shard, 0).num_rows == 5


def test_streaming_background_save_error_surfaces_and_finalizes(tmp_path, monkeypatch):
    from rlinf.data.storage.lerobot.writer import LeRobotDatasetWriter

    finalize_called = threading.Event()
    original_finalize = LeRobotDatasetWriter.finalize

    def fail_save(self, is_success=None):
        raise RuntimeError("synthetic save failure")

    def record_finalize(self):
        finalize_called.set()
        return original_finalize(self)

    monkeypatch.setattr(LeRobotDatasetWriter, "save_episode", fail_save)
    monkeypatch.setattr(LeRobotDatasetWriter, "finalize", record_finalize)

    env = CollectEpisode(
        _ScriptedEnv(
            [
                (np.array([False]), np.array([False]), {}),
                (np.array([True]), np.array([False]), {"success": np.array([True])}),
            ]
        ),
        save_dir=str(tmp_path / "collected"),
        num_envs=1,
        export_format="lerobot",
        robot_type="dual_yam",
        fps=30,
        streaming=True,
    )
    action = np.zeros((1, _STATE_DIM), dtype=np.float32)

    try:
        env.reset()
        env.step(action)
        env.step(action)
        env._wait_futures()
        assert finalize_called.wait(timeout=2.0)
        assert env._save_executor is not None
        # The finalize hook is set before the save future returns. Queue a direct
        # FIFO sentinel behind that save task so the future is done, while leaving
        # CollectEpisode's tracked future list intact for reset() to surface it.
        env._save_executor.submit(lambda: None).result(timeout=2.0)
        with pytest.raises(RuntimeError, match="synthetic save failure"):
            env.reset()
    finally:
        env.close()


def test_streaming_close_settles_frame_error_and_later_save_cleanup(
    tmp_path, monkeypatch
):
    from rlinf.data.storage.lerobot.writer import LeRobotDatasetWriter

    sidecar_calls = 0
    finalize_called = threading.Event()
    original_sidecar = CollectEpisode._write_stream_sidecar
    original_finalize = LeRobotDatasetWriter.finalize

    def fail_second_sidecar_write(self, writer, frame):
        nonlocal sidecar_calls
        sidecar_calls += 1
        if sidecar_calls == 2:
            raise RuntimeError("synthetic frame failure")
        return original_sidecar(self, writer, frame)

    def record_finalize(self):
        finalize_called.set()
        return original_finalize(self)

    monkeypatch.setattr(
        CollectEpisode, "_write_stream_sidecar", fail_second_sidecar_write
    )
    monkeypatch.setattr(LeRobotDatasetWriter, "finalize", record_finalize)

    save_dir = tmp_path / "collected"
    base_env = _ScriptedEnv(
        [
            (np.array([False]), np.array([False]), {}),
            (np.array([False]), np.array([False]), {}),
            (np.array([True]), np.array([False]), {"success": np.array([True])}),
        ]
    )
    env = CollectEpisode(
        base_env,
        save_dir=str(save_dir),
        num_envs=1,
        export_format="lerobot",
        robot_type="dual_yam",
        fps=30,
        streaming=True,
    )
    action = np.zeros((1, _STATE_DIM), dtype=np.float32)

    env.reset()
    env.step(action)
    env.step(action)
    env.step(action)
    with pytest.raises(RuntimeError, match="synthetic frame failure"):
        env.close()

    assert base_env.closed
    assert finalize_called.wait(timeout=1.0)

    shard = save_dir / "rank_0" / "id_0"
    info = json.loads((shard / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1
    assert _episode_frame_table(shard, 0).num_rows == 3


def test_streaming_mp4_export_does_not_block_recording_or_env_close(
    tmp_path, monkeypatch
):
    export_started = threading.Event()
    release_export = threading.Event()
    env_closed = threading.Event()
    export_released = threading.Event()
    export_calls = []

    def blocked_export(parquet_path, output_dir):
        export_calls.append((parquet_path, output_dir))
        export_started.set()
        release_export.wait(timeout=5.0)
        export_released.set()

    class CloseAwareEnv(_ScriptedEnv):
        def close(self):
            self.closed = True
            env_closed.set()

    monkeypatch.setattr(
        CollectEpisode, "_export_episode_mp4", staticmethod(blocked_export)
    )
    base_env = CloseAwareEnv(
        [
            (np.array([False]), np.array([False]), {}),
            (np.array([True]), np.array([False]), {"success": np.array([True])}),
            (np.array([False]), np.array([False]), {}),
        ]
    )
    env = CollectEpisode(
        base_env,
        save_dir=str(tmp_path / "collected"),
        num_envs=1,
        export_format="lerobot",
        robot_type="dual_yam",
        fps=30,
        streaming=True,
        export_mp4=True,
    )
    close_thread = None
    try:
        env.reset()
        action = np.zeros((1, _STATE_DIM), dtype=np.float32)
        env.step(action)
        env.step(action)
        assert export_started.wait(timeout=2.0), "MP4 export task was not submitted"
        assert len(export_calls) == 1

        _, _, _, _, info = env.step(action)
        assert not export_released.is_set()
        assert not bool(np.asarray(info["recording_invalid"]).any())

        close_thread = threading.Thread(target=env.close)
        close_thread.start()
        assert env_closed.wait(timeout=2.0), "env.close waited for MP4 export first"
        close_thread.join(timeout=0.2)
        assert close_thread.is_alive(), "close should wait for blocked MP4 export"
    finally:
        release_export.set()
        if close_thread is not None:
            close_thread.join(timeout=2.0)
        if close_thread is None or close_thread.is_alive():
            env.close()
    assert not (close_thread and close_thread.is_alive())
    assert base_env.closed


def test_streaming_requires_lerobot_format(tmp_path):
    with pytest.raises(ValueError, match="streaming"):
        CollectEpisode(
            _ScriptedEnv([]),
            save_dir=str(tmp_path / "x"),
            export_format="pickle",
            streaming=True,
        )
