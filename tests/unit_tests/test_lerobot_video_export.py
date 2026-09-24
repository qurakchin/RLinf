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

import importlib.util
import io
import json
from fractions import Fraction
from pathlib import Path

import pytest

_IMAGE_KEYS = ("image", "extra_view_image-0", "extra_view_image-1")
_VIEW_COLORS = {
    "image": (255, 0, 0),
    "extra_view_image-0": (0, 255, 0),
    "extra_view_image-1": (0, 0, 255),
}
_VIEW_OUTPUTS = {
    "image": "top.mp4",
    "extra_view_image-0": "left.mp4",
    "extra_view_image-1": "right.mp4",
}


@pytest.fixture(scope="module")
def video_export_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "toolkits"
        / "lerobot"
        / "visualize_lerobot_dataset.py"
    )
    spec = importlib.util.spec_from_file_location("_lerobot_video_export_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def deps():
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    av = pytest.importorskip("av")
    image_module = pytest.importorskip("PIL.Image")
    return pa, pq, av, image_module


def _dataset_info() -> dict:
    features = {
        key: {
            "dtype": "image",
            "shape": [64, 64, 3],
            "names": ["height", "width", "channel"],
        }
        for key in _IMAGE_KEYS
    }
    return {"robot_type": "dual_yam", "fps": 30, "features": features}


def _frame_color(image_key: str, frame_index: int) -> tuple[int, int, int]:
    variable = frame_index * 10
    if image_key == "image":
        return 200, variable, 20
    if image_key == "extra_view_image-0":
        return 20, 200, variable
    return variable, 20, 200


def _image_bytes(image_module, color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    image_module.new("RGB", (64, 64), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_episode_dataset(
    root: Path,
    *,
    pa,
    pq,
    image_module,
    episode_index: int = 0,
    published: bool = True,
    frame_count: int = 17,
    empty_key: str | None = None,
) -> tuple[Path, dict, dict[int, dict]]:
    meta_dir = root / "meta"
    parquet_dir = root / "data" / "chunk-000"
    meta_dir.mkdir(parents=True)
    parquet_dir.mkdir(parents=True)

    dataset_info = _dataset_info()
    (meta_dir / "info.json").write_text(json.dumps(dataset_info), encoding="utf-8")
    (meta_dir / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "pick_block"}) + "\n",
        encoding="utf-8",
    )
    if published:
        (meta_dir / "episodes.jsonl").write_text(
            json.dumps(
                {
                    "episode_index": episode_index,
                    "tasks": ["pick_block"],
                    "length": frame_count,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        (meta_dir / "episodes.jsonl").write_text("", encoding="utf-8")

    image_struct_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    arrays = []
    for key in _IMAGE_KEYS:
        rows = [
            {
                "bytes": (
                    b""
                    if key == empty_key
                    else _image_bytes(image_module, _frame_color(key, idx))
                ),
                "path": f"images/{key}/episode_{episode_index:06d}/frame_{idx:06d}.png",
            }
            for idx in range(frame_count)
        ]
        arrays.append(pa.array(rows, type=image_struct_type))

    parquet_path = parquet_dir / f"episode_{episode_index:06d}.parquet"
    pq.write_table(
        pa.Table.from_arrays(arrays, names=list(_IMAGE_KEYS)),
        parquet_path,
        row_group_size=5,
    )

    episode_meta_map = (
        {episode_index: {"episode_index": episode_index, "length": frame_count}}
        if published
        else {}
    )
    return parquet_path, dataset_info, episode_meta_map


def _export_episode(
    video_export_module, deps, parquet_path, output_dir, dataset_info, meta_map
):
    _, pq, av, image_module = deps
    return video_export_module._write_mp4_only_episode(
        parquet_path=parquet_path,
        output_dir=output_dir,
        dataset_info=dataset_info,
        episode_meta_map=meta_map,
        pq=pq,
        av=av,
        image_cls=image_module,
    )


def _decode_video(path: Path, av):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        duration_candidates = []
        if stream.duration is not None:
            duration_candidates.append(float(stream.duration * stream.time_base))
        if container.duration is not None:
            duration_candidates.append(float(container.duration * av.time_base))
        frames = [
            frame.to_ndarray(format="rgb24") for frame in container.decode(stream)
        ]
    return frames, fps, duration_candidates


def test_mp4_only_exports_three_dual_yam_views(video_export_module, deps, tmp_path):
    pa, _, av, image_module = deps
    parquet_path, dataset_info, meta_map = _write_episode_dataset(
        tmp_path / "dataset",
        pa=pa,
        pq=deps[1],
        image_module=image_module,
    )

    assert _export_episode(
        video_export_module,
        deps,
        parquet_path,
        tmp_path / "videos",
        dataset_info,
        meta_map,
    )

    episode_dir = tmp_path / "videos" / "episode_000000"
    parquet = deps[1].ParquetFile(parquet_path)
    assert parquet.num_row_groups == 4
    for image_key, output_name in _VIEW_OUTPUTS.items():
        frames, fps, durations = _decode_video(episode_dir / output_name, av)
        assert len(frames) == 17
        assert frames[0].shape == (64, 64, 3)
        assert fps == pytest.approx(30.0)
        assert Fraction(len(frames), round(fps)) == Fraction(17, 30)
        if durations:
            assert any(
                duration == pytest.approx(17 / 30, abs=0.05) for duration in durations
            )

        mean_rgb = sum(frame.mean(axis=(0, 1)) for frame in frames) / len(frames)
        dominant_channel = _VIEW_COLORS[image_key].index(255)
        assert mean_rgb[dominant_channel] > 160
        for channel in set(range(3)) - {dominant_channel}:
            assert mean_rgb[channel] < 90

        first_rgb = frames[0].mean(axis=(0, 1))
        last_rgb = frames[-1].mean(axis=(0, 1))
        variable_channel = (dominant_channel + 1) % 3
        assert first_rgb[dominant_channel] > 160
        assert last_rgb[dominant_channel] > 160
        assert last_rgb[variable_channel] > first_rgb[variable_channel] + 100

    assert not list((tmp_path / "videos").rglob("*.tmp.mp4"))


def test_mp4_only_empty_image_bytes_do_not_publish_failed_view(
    video_export_module, deps, tmp_path
):
    pa, _, _, image_module = deps
    parquet_path, dataset_info, meta_map = _write_episode_dataset(
        tmp_path / "dataset",
        pa=pa,
        pq=deps[1],
        image_module=image_module,
        empty_key="extra_view_image-1",
    )

    with pytest.raises(ValueError, match="Empty image bytes"):
        _export_episode(
            video_export_module,
            deps,
            parquet_path,
            tmp_path / "videos",
            dataset_info,
            meta_map,
        )

    episode_dir = tmp_path / "videos" / "episode_000000"
    assert (episode_dir / "top.mp4").is_file()
    assert (episode_dir / "left.mp4").is_file()
    assert not (episode_dir / "right.mp4").exists()
    assert not list((tmp_path / "videos").rglob("*.tmp.mp4"))


def test_mp4_only_skips_unpublished_episode(video_export_module, deps, tmp_path):
    pa, _, _, image_module = deps
    parquet_path, dataset_info, meta_map = _write_episode_dataset(
        tmp_path / "dataset",
        pa=pa,
        pq=deps[1],
        image_module=image_module,
        published=False,
    )

    assert not _export_episode(
        video_export_module,
        deps,
        parquet_path,
        tmp_path / "videos",
        dataset_info,
        meta_map,
    )
    assert not (tmp_path / "videos" / "episode_000000").exists()
    assert not list((tmp_path / "videos").rglob("*.mp4"))
