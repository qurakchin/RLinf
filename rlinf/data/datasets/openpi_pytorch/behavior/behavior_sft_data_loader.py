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

"""Self-contained BEHAVIOR-1K SFT data loader for the PyTorch pi05 path.

Ports ``rlinf/models/embodiment/openpi/dataconfig/behavior_data_loader.py`` onto
the vendored ``openpi_pytorch`` primitives, with zero installed-``openpi``
imports. The loader streams the BEHAVIOR dataset
(:class:`~.behavior_sft_dataset.BehaviorSftDataset`), applies the per-sample
:func:`~.behavior_sft_transform.BehaviorSftTransform`, collates samples into a
batched :class:`Observation` plus an actions tensor of shape
``[batch, action_horizon, action_dim]``, and yields ``(Observation, actions)``.

The streaming dataset partitions its keyframe chunks per ``(rank, worker)``
internally (see :meth:`BehaviorSftDataset.__getitem__`), so a
``DistributedSampler`` is intentionally *not* used: a sampler only reorders the
ignored ``idx`` values and would otherwise give every distributed rank identical
data.
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing
import pathlib
import typing

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.data.datasets.behavior.behavior_sft_dataset import (
    BehaviorSftDataset,
)
from rlinf.data.datasets.behavior.behavior_sft_transform import (
    BehaviorSftTransform,
    transform_behavior_sft_item,
)
from rlinf.data.lerobot_paths import (
    resolve_lerobot_dataset_root,
    resolve_lerobot_repo_id,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_pytorch.utils.normalize import (
    NormStats,
    blank_asset_field,
    load_norm_stats,
    resolve_norm_stats_dir,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BehaviorSftDataConfig",
    "build_behavior_sft_dataloader",
    "collate_behavior_sft_items",
    "create_behavior_sft_data_loader",
    "transform_behavior_sft_item",
]

# Default BEHAVIOR pi05 dataset repo (matches the old LeRobotB1KDataConfig).
_DEFAULT_REPO_ID = "behavior-1k/2025-challenge-demos"
# Camera views resolved by the BEHAVIOR pi05 transform.
_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


@dataclasses.dataclass(frozen=True)
class BehaviorSftDataConfig:
    """Metadata describing the BEHAVIOR SFT data pipeline.

    Exposed via :meth:`BehaviorSftDataLoader.data_config` so the SFT worker can
    read the resolved repo id, action dimension, action horizon, and the
    normalization statistics without reaching into the dataset internals.
    """

    repo_id: str
    action_dim: int
    action_horizon: int
    max_token_len: int
    norm_stats: dict[str, NormStats]


class _TransformedStreamingDataset(torch.utils.data.Dataset):
    """Wrap the streaming dataset, applying the per-sample SFT transform.

    The transform holds a (non-picklable) SentencePiece tokenizer; it is built
    lazily inside each ``spawn`` worker on first use, so only the lightweight
    :class:`BehaviorSftTransform` config travels across the process boundary.
    """

    def __init__(self, dataset: BehaviorSftDataset, transform: BehaviorSftTransform):
        self._dataset = dataset
        self._transform = transform

    def __getitem__(self, idx):
        frame = self._dataset[idx]
        return transform_behavior_sft_item(frame, self._transform)

    def __len__(self) -> int:
        # The streaming dataset ignores `idx` and partitions chunks internally;
        # `len` only drives torch's default index sampler so iteration proceeds.
        return len(self._dataset.hf_dataset)


def collate_behavior_sft_items(
    items: typing.Sequence[typing.Mapping[str, typing.Any]],
) -> tuple[Observation, torch.Tensor]:
    """Collate transformed items into ``(Observation, actions)``.

    Images are stacked as ``uint8`` ``[B, H, W, C]`` tensors and converted to
    ``float32`` in ``[-1, 1]`` by :meth:`Observation.from_dict` (matching the old
    path). State/actions/tokens are stacked into the appropriate torch dtypes;
    the returned actions tensor has shape ``[batch, action_horizon, action_dim]``.
    """
    if not items:
        raise ValueError("Cannot collate an empty BEHAVIOR SFT batch.")

    images = {
        key: torch.from_numpy(
            np.stack([np.asarray(item["image"][key]) for item in items])
        )
        for key in _IMAGE_KEYS
    }
    image_masks = {
        key: torch.from_numpy(
            np.stack(
                [np.asarray(item["image_mask"][key], dtype=np.bool_) for item in items]
            )
        )
        for key in _IMAGE_KEYS
    }
    batch = {
        "image": images,
        "image_mask": image_masks,
        "state": torch.from_numpy(
            np.stack([np.asarray(item["state"], dtype=np.float32) for item in items])
        ),
        "tokenized_prompt": torch.from_numpy(
            np.stack(
                [np.asarray(item["tokenized_prompt"], dtype=np.int64) for item in items]
            )
        ).long(),
        "tokenized_prompt_mask": torch.from_numpy(
            np.stack(
                [
                    np.asarray(item["tokenized_prompt_mask"], dtype=np.bool_)
                    for item in items
                ]
            )
        ),
    }
    actions = torch.from_numpy(
        np.stack([np.asarray(item["actions"], dtype=np.float32) for item in items])
    )
    return Observation.from_dict(batch), actions


def _worker_init_fn(worker_id: int) -> None:
    """Per-worker init hook (placeholder for worker-local environment setup)."""
    del worker_id


def _collate_id_items(items):
    """Audit-only collate: stack the value-independent (episode_index, frame_index) ids.
    Module-level so spawned DataLoader workers can pickle it."""
    return [(it["episode_index"], it["frame_index"]) for it in items]


def _resolve_norm_stats(
    assets_dir: str | pathlib.Path,
    asset_id: str | None,
) -> dict[str, NormStats]:
    """Resolve the BEHAVIOR norm stats via the shared ``(assets_dir, asset_id)`` rule.

    Delegates to :func:`resolve_norm_stats_dir` so the SFT loader and the eval
    model factory resolve the *same* canonical ``norm_stats.json``.
    """
    directory = resolve_norm_stats_dir(assets_dir, asset_id)
    logger.info("Loaded BEHAVIOR norm stats from %s", directory)
    return load_norm_stats(directory)


def create_behavior_sft_data_loader(
    *,
    behavior_dataset_root: str,
    assets_dir: str,
    asset_id: str | None,
    repo_id: str = _DEFAULT_REPO_ID,
    tasks: list[str] | None = None,
    modalities: list[str] | None = None,
    action_dim: int = 32,
    action_horizon: int = 32,
    max_token_len: int = 200,
    batch_size: int = 1,
    num_workers: int = 0,
    fine_grained_level: int = 0,
    tolerance_s: float = 1e-4,
    shuffle: bool = True,
    skip_norm_stats: bool = False,
    seed: int = 0,
    norm_stats: dict[str, NormStats] | None = None,
    skill_labels: dict[int, str] | None = None,
    use_skill: bool = False,
    enable_gap: bool = True,
    allow_left: int = 0,
    allow_right: int = 0,
    dist_rank: int | None = None,
    dist_world_size: int | None = None,
    id_only: bool = False,
) -> "BehaviorSftDataLoader":
    """Build the BEHAVIOR-1K SFT data loader yielding ``(Observation, actions)``.

    Args:
        behavior_dataset_root: Local root of the LeRobot BEHAVIOR dataset.
        assets_dir: Directory holding the checkpoint assets (norm stats).
        asset_id: Sub-directory under ``assets_dir`` for the norm stats
            (``{assets_dir}/{asset_id}/norm_stats.json``).
        repo_id: LeRobot dataset repo id (used for metadata bookkeeping).
        tasks: BEHAVIOR task names to include (``None`` -> all tasks).
        modalities: Observation modalities to load (``None`` -> ``["rgb"]``).
        action_dim: Model action dimension to pad state/actions to.
        action_horizon: Number of future action steps per sample.
        max_token_len: Maximum tokenized-prompt length.
        batch_size: Per-rank batch size.
        num_workers: Number of ``DataLoader`` workers (``> 0`` uses ``spawn``).
        fine_grained_level: Orchestrator level for the prompt task text.
        tolerance_s: Frame-timestamp sync tolerance.
        shuffle: Whether the streaming dataset shuffles its chunk order.
        skip_norm_stats: Skip loading norm stats (requires ``norm_stats`` to be
            supplied if normalization is still desired).
        seed: Base seed for the streaming chunk partition.
        norm_stats: Pre-loaded norm stats; loaded from disk when ``None``.
        skill_labels: Optional per-skill labels enabling skill mode.
        use_skill: Train on per-frame SKILL text (window-resolved) instead of the
            main-task text; requires explicit ``skill_labels`` (raises otherwise).
        enable_gap: Skill mode — absorb a true gap into both adjacent skills.
        allow_left: Skill mode — frames to extend a contiguous skill start left.
        allow_right: Skill mode — frames to extend a contiguous skill end right.

    Returns:
        A loader whose iteration yields ``(Observation, actions)`` 2-tuples.
    """
    if norm_stats is None and not skip_norm_stats:
        norm_stats = _resolve_norm_stats(assets_dir, asset_id)
    elif norm_stats is None:
        raise ValueError(
            "norm_stats must be provided when skip_norm_stats=True so the state "
            "and actions can still be quantile-normalized."
        )

    repo_id = repo_id or _DEFAULT_REPO_ID

    dataset = BehaviorSftDataset(
        repo_id=repo_id,
        root=behavior_dataset_root,
        tolerance_s=tolerance_s,
        tasks=tasks or None,
        modalities=modalities or ["rgb"],
        local_only=True,
        delta_timestamps={"action": [t / 30.0 for t in range(action_horizon)]},
        chunk_streaming_using_keyframe=True,
        shuffle=shuffle,
        seed=seed,
        fine_grained_level=fine_grained_level,
        skill_labels=skill_labels,
        use_skill=use_skill,
        enable_gap=enable_gap,
        allow_left=allow_left,
        allow_right=allow_right,
        dist_rank=dist_rank,
        dist_world_size=dist_world_size,
        id_only=id_only,
    )

    if id_only:
        # Audit-only: the dataset yields value-independent {episode_index, frame_index}
        # ids (no transform/normalization/tokenization). Collate stacks the ids; the rest
        # of the production topology (partition, num_workers, drop_last, seed) is kept.
        # The collate must be a module-level function so spawned workers can pickle it.
        source = dataset
        collate = _collate_id_items
    else:
        transform = BehaviorSftTransform(
            norm_stats=norm_stats,
            action_dim=action_dim,
            max_token_len=max_token_len,
        )
        source = _TransformedStreamingDataset(dataset, transform)
        collate = collate_behavior_sft_items

    # The streaming dataset partitions chunks per (rank, worker) on its own, so a
    # DistributedSampler is intentionally omitted: it would only reorder the
    # ignored `idx` values and give every distributed rank identical data.
    mp_context = multiprocessing.get_context("spawn") if num_workers > 0 else None

    generator = torch.Generator()
    generator.manual_seed(seed)

    logger.info(
        "BEHAVIOR SFT data loader: batch_size=%d, num_workers=%d, action_horizon=%d",
        batch_size,
        num_workers,
        action_horizon,
    )

    torch_loader = torch.utils.data.DataLoader(
        typing.cast(torch.utils.data.Dataset, source),
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=None,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
        collate_fn=collate,
        worker_init_fn=_worker_init_fn,
        drop_last=True,
        generator=generator,
    )

    data_config = BehaviorSftDataConfig(
        repo_id=repo_id,
        action_dim=action_dim,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
        norm_stats=norm_stats,
    )
    return BehaviorSftDataLoader(torch_loader, data_config)


class BehaviorSftDataLoader:
    """Infinite ``(Observation, actions)`` loop over the BEHAVIOR SFT dataset.

    Mirrors the old behavior data loader: it re-iterates the underlying ``torch``
    ``DataLoader`` forever. Each batch is already collated into an
    :class:`Observation` plus an actions tensor of shape
    ``[batch, action_horizon, action_dim]`` by :func:`collate_behavior_sft_items`.
    """

    def __init__(
        self,
        torch_loader: torch.utils.data.DataLoader,
        data_config: BehaviorSftDataConfig,
    ):
        self._torch_loader = torch_loader
        self._data_config = data_config

    def data_config(self) -> BehaviorSftDataConfig:
        """Return the resolved data-pipeline metadata."""
        return self._data_config

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        """Expose the underlying ``torch`` ``DataLoader``."""
        return self._torch_loader

    def __iter__(self):
        while True:
            yield from self._torch_loader

    def __len__(self) -> int:
        return len(self._torch_loader)


PER_RANK_STREAM = "per_rank_stream"
REFERENCE_FANOUT = "reference_fanout"
_LOADER_MODES = (PER_RANK_STREAM, REFERENCE_FANOUT)


def resolve_loader_mode(cfg, loader_mode=None):
    """Resolve the BEHAVIOR SFT loading strategy (validated).

    ``per_rank_stream`` (default): every rank builds its own loader and streams a
    rank-folded chunk shard (``world_size * num_workers`` lanes) -- the throughput
    path. ``reference_fanout``: a single loader with a worker-only chunk partition
    (``num_workers`` lanes, NO rank fold) is pulled ``world_size`` micro-batches per
    step and scattered, byte-reproducing the reference ``openpi-comet`` rank-0 fanout
    stream (including the epoch boundary). Selected via ``data.loader_mode``.
    """
    mode = loader_mode
    if mode is None:
        mode = str(OmegaConf.select(cfg.data, "loader_mode", default=PER_RANK_STREAM))
    if mode not in _LOADER_MODES:
        raise ValueError(
            f"data.loader_mode must be one of {_LOADER_MODES}; got {mode!r}."
        )
    return mode


def builds_train_loader(loader_mode, rank):
    """Whether ``rank`` should construct + iterate the TRAIN loader.

    ``per_rank_stream``: every rank owns its shard. ``reference_fanout``: only rank 0 owns
    the single loader and scatters to the others, so ranks ``> 0`` must NOT build/iterate a
    train loader they never consume (they receive via the fanout).
    """
    return loader_mode != REFERENCE_FANOUT or rank == 0


def reference_fanout_micro_batches(
    data_iter, rank, world_size, grad_accum, send_fn, recv_fn
):
    """Return THIS rank's ``grad_accum`` micro-batches for one step via rank-0 fanout.

    Byte-reproduces the reference ``openpi-comet`` ``train_pytorch_new.py`` loop: rank 0
    owns the single (worker-only-partition) loader and, PER STEP, pulls ``grad_accum``
    micro-batches for each rank ``1..world_size-1`` (in order) and sends them, then pulls
    its OWN ``grad_accum`` micro-batches LAST; every other rank receives its micro-batches
    from rank 0. The pull order (others first, rank 0 last) is what makes each rank's data
    identical to the reference, not merely the global multiset.

    ``send_fn(batches, dst)`` / ``recv_fn(src)`` are injected so the pure control flow is
    unit-testable with a fake loader + a CPU process group (the production worker passes
    ``torch.distributed`` ``send_object_list`` / ``recv_object_list`` wrappers).
    """
    if rank == 0:
        for dst in range(1, world_size):
            send_fn([next(data_iter) for _ in range(grad_accum)], dst)
        return [next(data_iter) for _ in range(grad_accum)]
    return recv_fn(0)


def build_behavior_sft_dataloader(
    cfg, world_size, rank, data_paths, eval_dataset=False, id_only=False, loader_mode=None
):
    """Build the self-contained BEHAVIOR SFT data loader for the SFT worker.

    Owns the config extraction the FSDP SFT worker previously did inline. In the
    default ``per_rank_stream`` mode the streaming dataset partitions chunks per
    ``(rank, worker)``; ``rank``/``world_size`` are captured here (in the main
    process) and threaded into the dataset so that SPAWNED DataLoader workers --
    which cannot read ``torch.distributed`` -- still partition by the correct
    per-rank id (otherwise every rank replicates rank 0's chunks, collapsing the
    effective batch to one rank's micro-batch).

    In ``reference_fanout`` mode the dataset is partitioned by WORKER only
    (``dist_rank=0``, ``dist_world_size=1``) so its ``num_workers`` lanes match the
    reference's single rank-0 loader; the caller pulls ``world_size`` micro-batches
    per step and scatters them (see the SFT worker). Returns
    ``(loader, loader.data_config())``.
    """
    # reference_fanout is a TRAIN-only strict-alignment mode; eval always uses the
    # decentralized per-rank shard so it is not silently made rank-replicated.
    mode = PER_RANK_STREAM if eval_dataset else resolve_loader_mode(cfg, loader_mode)
    # reference_fanout: a single worker-only-partition loader (no rank fold), pulled
    # world_size micro-batches/step downstream, reproduces the reference rank-0 fanout.
    part_rank, part_world = (0, 1) if mode == REFERENCE_FANOUT else (rank, world_size)

    data_path = resolve_lerobot_repo_id(data_paths)
    if data_path is None:
        raise ValueError("openpi_pytorch BEHAVIOR SFT requires data.train_data_paths.")

    model_cfg = cfg.actor.model
    openpi_data = OmegaConf.select(cfg.actor, "openpi_data", default={})
    if not isinstance(openpi_data, DictConfig):
        openpi_data = OmegaConf.create(openpi_data)

    def model_select(key, default):
        return OmegaConf.select(model_cfg, key, default=default)

    def data_select(key, default):
        return OmegaConf.select(openpi_data, key, default=default)

    def data_cfg_select(key, default):
        return OmegaConf.select(cfg.data, key, default=default)

    # Norm stats are resolved STRICTLY from YAML assets_dir + asset_id — the same
    # canonical task-0000 distribution the eval model factory resolves.
    # No checkpoint-relative (model_path) or norm_stats_path fallback, and a blank
    # (None or empty/whitespace) value is rejected here the same way the eval
    # factory rejects it, so neither path can silently load non-task-0000 stats.
    assets_dir = model_select("openpi.assets_dir", None)
    asset_id = model_select("openpi.asset_id", None)
    missing = blank_asset_field(assets_dir, asset_id)
    if missing is not None:
        raise ValueError(
            f"openpi_pytorch BEHAVIOR SFT requires a non-empty "
            f"actor.model.openpi.{missing} (the canonical task-0000 norm-stats "
            "asset location); it has no default."
        )

    micro_batch_size = cfg.actor.micro_batch_size
    eval_batch_size = cfg.actor.get("eval_batch_size", 1)

    # `cfg.data` is the production source of truth for the BEHAVIOR task set and the
    # prompt-source flag. `use_skill: true` trains on the per-frame REFERENCE skill
    # text; `false` trains on the main-task text.
    use_skill = bool(data_cfg_select("use_skill", False))
    tasks = list(data_cfg_select("tasks", ["turning_on_radio"]))
    skill_labels, enable_gap, allow_left, allow_right = None, True, 0, 0
    if use_skill:
        # The skill labels are the REFERENCE per-task subtask list from config (NOT
        # the dataset's collapsed orchestrators, which equal the full task text). The
        # task-0000 local-skill recipe is exactly one task with a configured subtask
        # list and the fixed window recipe below.
        if len(tasks) != 1:
            raise ValueError(
                "openpi_pytorch BEHAVIOR SFT use_skill:true supports exactly one task "
                f"(the task-0000 skill recipe); got data.tasks={tasks}."
            )
        subtask_labels = data_cfg_select("task_subtasks", {})
        labels = subtask_labels.get(tasks[0]) if subtask_labels else None
        if not labels:
            raise ValueError(
                "openpi_pytorch BEHAVIOR SFT use_skill:true requires the reference "
                f"skill labels at data.task_subtasks.{tasks[0]}; none was configured."
            )
        skill_labels = {i: str(label) for i, label in enumerate(labels)}
        # Fixed reference skill-window recipe (pi05_b1k-task0000_sft_local_skill);
        # not read from cfg.data so the reference recipe cannot drift.
        enable_gap, allow_left, allow_right = True, 100, 100

    loader = create_behavior_sft_data_loader(
        behavior_dataset_root=str(
            data_select(
                "behavior_dataset_root",
                resolve_lerobot_dataset_root(str(data_path)),
            )
        ),
        assets_dir=str(assets_dir),
        asset_id=asset_id,
        repo_id=str(data_select("repo_id", _DEFAULT_REPO_ID)),
        tasks=tasks,
        modalities=list(data_select("modalities", ["rgb"])),
        action_dim=int(model_select("openpi.model_action_dim", 32)),
        action_horizon=int(model_select("num_action_chunks", 32)),
        max_token_len=int(model_select("openpi.max_token_len", 200)),
        batch_size=eval_batch_size if eval_dataset else micro_batch_size,
        num_workers=int(data_cfg_select("num_workers", 8)),
        fine_grained_level=int(data_select("fine_grained_level", 0)),
        tolerance_s=float(data_select("tolerance_s", 1e-4)),
        shuffle=not eval_dataset,
        seed=int(cfg.actor.get("seed", 42)),
        skill_labels=skill_labels,
        use_skill=use_skill,
        enable_gap=enable_gap,
        allow_left=allow_left,
        allow_right=allow_right,
        dist_rank=part_rank,
        dist_world_size=part_world,
        id_only=id_only,
    )
    return loader, loader.data_config()
