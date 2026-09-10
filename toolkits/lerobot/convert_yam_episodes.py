#!/usr/bin/env python3
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

"""Convert canonical YAM episodes to RLinf's LeRobot collection schema.

The source directory is left untouched. Data is first written beside the final
output as ``<output>.partial`` and is renamed only after every episode has been
saved successfully.

Example::

    .venv/bin/python toolkits/lerobot/convert_yam_episodes.py \
        /home/yambox/RLinf_yam/data/episodes/pick_block \
        /home/yambox/RLinf_yam/data/lerobot/pick_block

The output matches ``CollectEpisode(export_format="lerobot")`` for dual YAM:
``state`` and ``actions`` use the fixed
``[left_q0..q5, left_gripper, right_q0..q5, right_gripper]`` order; ``top`` is
stored as ``image`` and the remaining camera roles are stored as
``extra_view_image-0``, ``extra_view_image-1``, and so on.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.data.storage.lerobot import add_frame_to_dataset
from rlinf.data.storage.lerobot.writer import LeRobotDatasetWriter

WRITE_COMPLETE_FLAG = "write_complete.flag"
EXPECTED_ARMS = ("left", "right")


def nearest_indices(reference: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Return the nearest source index for every reference timestamp.

    Args:
        reference: One-dimensional reference timestamps.
        source: Sorted one-dimensional source timestamps.

    Returns:
        A monotonically nondecreasing integer index array.

    Raises:
        ValueError: If either timestamp array is invalid or empty.
    """
    reference = np.asarray(reference)
    source = np.asarray(source)
    if reference.ndim != 1 or source.ndim != 1:
        raise ValueError("camera timestamps must be one-dimensional")
    if reference.size == 0 or source.size == 0:
        raise ValueError("camera timestamps must not be empty")
    if np.any(np.diff(source) < 0):
        raise ValueError("source camera timestamps must be sorted")
    if source.size == 1:
        return np.zeros(reference.size, dtype=np.int64)
    right = np.clip(np.searchsorted(source, reference), 1, source.size - 1)
    choose_left = (reference - source[right - 1]) <= (source[right] - reference)
    return (right - choose_left.astype(np.int64)).astype(np.int64, copy=False)


@dataclass(frozen=True)
class CameraSpec:
    """One camera stream described by a YAM episode's metadata."""

    role: str
    height: int
    width: int


@dataclass(frozen=True)
class Episode:
    """Validated paths and metadata for one source episode."""

    path: Path
    metadata: dict[str, Any]
    cameras: tuple[CameraSpec, ...]
    frame_count: int


class VideoReader:
    """Read monotonically increasing MP4 indices without buffering a video."""

    def __init__(self, path: Path, expected_frames: int) -> None:
        import cv2

        self._cv2 = cv2
        self._path = path
        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            raise RuntimeError(f"failed to open video: {path}")
        actual_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if actual_frames != expected_frames:
            self.close()
            raise ValueError(
                f"{path}: video has {actual_frames} frames, timestamps have "
                f"{expected_frames}"
            )
        self._current_index = -1
        self._current_rgb: np.ndarray | None = None

    def read(self, index: int) -> np.ndarray:
        """Decode and return one RGB frame at a nondecreasing index."""
        if index < self._current_index:
            raise ValueError(
                f"video indices must be nondecreasing: {index} < {self._current_index}"
            )
        while self._current_index < index:
            ok, bgr = self._capture.read()
            if not ok or bgr is None:
                raise RuntimeError(
                    f"{self._path}: decode failed at frame {self._current_index + 1}"
                )
            self._current_index += 1
            self._current_rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        if self._current_rgb is None:
            raise RuntimeError(f"{self._path}: no frame was decoded")
        return self._current_rgb

    def close(self) -> None:
        """Release the underlying decoder."""
        self._capture.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def _completed_episode_paths(source: Path) -> list[Path]:
    if (source / WRITE_COMPLETE_FLAG).is_file():
        return [source]
    return sorted(
        path
        for path in source.iterdir()
        if path.is_dir() and (path / WRITE_COMPLETE_FLAG).is_file()
    )


def _load_episode(path: Path, main_camera: str) -> Episode:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    arms = tuple(metadata.get("arm_names", ()))
    if arms != EXPECTED_ARMS:
        raise ValueError(f"{path}: expected arm_names {EXPECTED_ARMS}, got {arms}")
    if int(metadata.get("num_arm_joints", -1)) != 6:
        raise ValueError(f"{path}: dual YAM data must have six joints per arm")
    frame_count = int(metadata.get("num_frames", 0))
    if frame_count <= 0:
        raise ValueError(f"{path}: num_frames must be positive")

    cameras = tuple(
        CameraSpec(
            role=str(camera["role"]),
            height=int(camera["height"]),
            width=int(camera["width"]),
        )
        for camera in metadata.get("cameras", ())
    )
    roles = [camera.role for camera in cameras]
    if main_camera not in roles:
        raise ValueError(
            f"{path}: main camera {main_camera!r} is absent; found {roles}"
        )
    if len(set(roles)) != len(roles):
        raise ValueError(f"{path}: duplicate camera roles: {roles}")

    expected_arrays = []
    for arm in EXPECTED_ARMS:
        expected_arrays.extend(
            (
                (f"{arm}-joint_pos.npy", (frame_count, 6)),
                (f"{arm}-gripper_pos.npy", (frame_count, 1)),
                (f"action-{arm}-joint.npy", (frame_count, 6)),
                (f"action-{arm}-gripper.npy", (frame_count, 1)),
            )
        )
    for filename, shape in expected_arrays:
        array_path = path / filename
        if not array_path.is_file():
            raise FileNotFoundError(f"missing source array: {array_path}")
        actual_shape = np.load(array_path, mmap_mode="r").shape
        if actual_shape != shape:
            raise ValueError(
                f"{array_path}: expected shape {shape}, got {actual_shape}"
            )
    for camera in cameras:
        timestamp_path = path / f"{camera.role}-timestamp.npy"
        video_path = path / f"{camera.role}-images-rgb.mp4"
        if not timestamp_path.is_file():
            raise FileNotFoundError(f"missing camera timestamps: {timestamp_path}")
        if not video_path.is_file():
            raise FileNotFoundError(f"missing camera video: {video_path}")
    return Episode(path, metadata, cameras, frame_count)


def _load_vectors(episode: Episode) -> tuple[np.ndarray, np.ndarray]:
    state_parts = []
    action_parts = []
    for arm in EXPECTED_ARMS:
        state_parts.extend(
            (
                np.load(episode.path / f"{arm}-joint_pos.npy", mmap_mode="r"),
                np.load(episode.path / f"{arm}-gripper_pos.npy", mmap_mode="r"),
            )
        )
        action_parts.extend(
            (
                np.load(episode.path / f"action-{arm}-joint.npy", mmap_mode="r"),
                np.load(episode.path / f"action-{arm}-gripper.npy", mmap_mode="r"),
            )
        )
    return (
        np.concatenate(state_parts, axis=1).astype(np.float32),
        np.concatenate(action_parts, axis=1).astype(np.float32),
    )


def _ordered_cameras(
    cameras: tuple[CameraSpec, ...], main_camera: str
) -> tuple[CameraSpec, ...]:
    main = next(camera for camera in cameras if camera.role == main_camera)
    return (main, *(camera for camera in cameras if camera.role != main_camera))


def _release_saved_episode(dataset: Any) -> None:
    """Drop LeRobot's cumulative in-memory table after an episode is saved.

    LeRobot 0.1 appends every saved episode to ``hf_dataset`` even though each
    episode has already been persisted as its own parquet file.  With embedded
    image features that makes conversion memory grow with the entire dataset.
    Replacing the table with an empty, schema-compatible dataset keeps the
    writer metadata and on-disk files intact while bounding memory to roughly
    one episode.
    """
    if not hasattr(dataset, "hf_dataset"):
        return
    create_hf_dataset = getattr(dataset, "create_hf_dataset", None)
    if not callable(create_hf_dataset):
        raise RuntimeError(
            "LeRobot exposes hf_dataset but cannot create an empty replacement"
        )

    saved_dataset = dataset.hf_dataset
    dataset.hf_dataset = create_hf_dataset()
    del saved_dataset
    gc.collect()

    try:
        import pyarrow as pa

        pa.default_memory_pool().release_unused()
    except (AttributeError, ImportError):
        pass


def _validate_collection(episodes: list[Episode], main_camera: str) -> None:
    first = episodes[0]
    first_cameras = _ordered_cameras(first.cameras, main_camera)
    first_shapes = [(cam.role, cam.height, cam.width) for cam in first_cameras]
    first_hz = float(first.metadata["control_hz"])
    for episode in episodes[1:]:
        cameras = _ordered_cameras(episode.cameras, main_camera)
        shapes = [(cam.role, cam.height, cam.width) for cam in cameras]
        if shapes != first_shapes:
            raise ValueError(
                f"{episode.path}: camera schema {shapes} differs from {first_shapes}"
            )
        if float(episode.metadata["control_hz"]) != first_hz:
            raise ValueError(
                f"{episode.path}: control_hz differs from the first episode"
            )


def _add_episode(
    dataset: Any,
    episode: Episode,
    main_camera: str,
    task: str,
    is_success: bool,
) -> None:
    state, actions = _load_vectors(episode)
    cameras = _ordered_cameras(episode.cameras, main_camera)
    reference_ts = np.load(episode.path / f"{main_camera}-timestamp.npy", mmap_mode="r")
    camera_indices = {}
    for camera in cameras:
        timestamps = np.load(
            episode.path / f"{camera.role}-timestamp.npy", mmap_mode="r"
        )
        camera_indices[camera.role] = nearest_indices(reference_ts, timestamps)

    with ExitStack() as stack:
        readers = {
            camera.role: stack.enter_context(
                VideoReader(
                    episode.path / f"{camera.role}-images-rgb.mp4",
                    len(
                        np.load(
                            episode.path / f"{camera.role}-timestamp.npy",
                            mmap_mode="r",
                        )
                    ),
                )
            )
            for camera in cameras
        }
        for index in range(episode.frame_count):
            frame: dict[str, Any] = {
                "state": state[index],
                "actions": actions[index],
                "task": task,
                "done": np.array([index == episode.frame_count - 1], dtype=bool),
                "is_success": np.array([is_success], dtype=bool),
                "intervene_flag": np.array([True], dtype=bool),
                "segment_id": np.array([0], dtype=np.uint8),
            }
            for camera_index, camera in enumerate(cameras):
                feature = (
                    "image"
                    if camera_index == 0
                    else f"extra_view_image-{camera_index - 1}"
                )
                source_index = int(camera_indices[camera.role][index])
                frame[feature] = readers[camera.role].read(source_index)
            add_frame_to_dataset(dataset, frame)
    dataset.save_episode()
    _release_saved_episode(dataset)


def convert(args: argparse.Namespace) -> None:
    """Validate and convert the collection described by command-line arguments."""
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"output already exists; refusing to overwrite: {output}")
    partial = output.with_name(f"{output.name}.partial")
    if partial.exists():
        raise FileExistsError(
            f"partial output already exists; inspect or move it before retrying: {partial}"
        )

    paths = _completed_episode_paths(source)
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"no completed episodes found under {source}")
    episodes = [_load_episode(path, args.main_camera) for path in paths]
    _validate_collection(episodes, args.main_camera)
    total_frames = sum(episode.frame_count for episode in episodes)
    print(
        f"Validated {len(episodes)} episodes ({total_frames} frames) from {source}",
        flush=True,
    )
    if args.dry_run:
        return

    first = episodes[0]
    cameras = _ordered_cameras(first.cameras, args.main_camera)
    fps = max(1, int(round(float(first.metadata["control_hz"]))))
    task = args.task or str(first.metadata.get("task_name") or source.name)
    extra_keys = {
        f"extra_view_image-{index}": (camera.height, camera.width, 3)
        for index, camera in enumerate(cameras[1:])
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = LeRobotDatasetWriter()
    started_at = time.monotonic()
    try:
        writer.create(
            repo_id=str(partial),
            robot_type="dual_yam",
            fps=fps,
            image_shape=(cameras[0].height, cameras[0].width, 3),
            state_dim=14,
            action_dim=14,
            extra_view_image_keys=extra_keys,
            has_intervene_flag=True,
            has_segment_id=True,
            image_writer_processes=0,
            image_writer_threads=args.image_writer_threads,
        )
        for output_index, episode in enumerate(episodes):
            episode_started = time.monotonic()
            _add_episode(
                writer.dataset,
                episode,
                args.main_camera,
                task,
                args.is_success,
            )
            elapsed = time.monotonic() - episode_started
            print(
                f"[{output_index + 1}/{len(episodes)}] {episode.path.name}: "
                f"{episode.frame_count} frames in {elapsed:.1f}s",
                flush=True,
            )
        writer.finalize()
    except BaseException:
        if writer.dataset is not None:
            writer.finalize()
        raise

    manifest = {
        "source": str(source),
        "output_schema": "RLinf CollectEpisode lerobot v2.1",
        "robot_type": "dual_yam",
        "fps": fps,
        "task": task,
        "is_success": args.is_success,
        "intervene_flag": True,
        "state_action_order": [
            "left_q0",
            "left_q1",
            "left_q2",
            "left_q3",
            "left_q4",
            "left_q5",
            "left_gripper",
            "right_q0",
            "right_q1",
            "right_q2",
            "right_q3",
            "right_q4",
            "right_q5",
            "right_gripper",
        ],
        "camera_features": {
            camera.role: ("image" if index == 0 else f"extra_view_image-{index - 1}")
            for index, camera in enumerate(cameras)
        },
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "source_episodes": [episode.path.name for episode in episodes],
    }
    (partial / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    os.rename(partial, output)
    elapsed = time.monotonic() - started_at
    print(f"Finished: {output} ({elapsed / 60:.1f} minutes)", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="YAM task episode directory")
    parser.add_argument("output", type=Path, help="new LeRobot dataset directory")
    parser.add_argument(
        "--main-camera",
        default="top",
        help="camera role stored as RLinf's main 'image' feature (default: top)",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="task text override (default: source metadata task_name)",
    )
    parser.add_argument(
        "--is-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="mark converted episodes successful (default: true)",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=8,
        help="LeRobot image writer thread count (default: 8)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="convert only the first N completed episodes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate metadata and arrays without creating output",
    )
    return parser


def main() -> None:
    """Run the command-line converter."""
    args = _parser().parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.image_writer_threads < 0:
        raise SystemExit("--image-writer-threads must be nonnegative")
    try:
        convert(args)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
