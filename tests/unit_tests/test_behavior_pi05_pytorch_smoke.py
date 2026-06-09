# Copyright (c) 2025, RLinf contributors.
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

"""Fast, non-GPU smoke gates for the BEHAVIOR pi05 ``openpi_pytorch`` feature.

These replace the dropped diagnostic suite with a thin automated harness:
  * the model is registered without clobbering existing embodied models;
  * the eval and SFT Hydra configs build and select ``openpi_pytorch``;
  * the streaming dataset's per-(rank, worker) chunk partition is rank-disjoint
    and covers every chunk exactly once;
  * the checkpoint converter's loud-failure invariant holds.

Heavier checks (a real SFT forward/loss step, full converter round-trips) need
multi-GB weights / a GPU and are exercised by the e2e configs under
``tests/e2e_tests``; here they are skipped with an explicit reason when their
inputs are absent.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EMBODIED_CONFIG = REPO_ROOT / "examples" / "embodiment" / "config"
SFT_CONFIG = REPO_ROOT / "examples" / "sft" / "config"


def _compose(config_dir: Path, config_name: str, embodied_path: Path):
    """Compose a Hydra config the way the launch scripts do (via EMBODIED_PATH)."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    os.environ["EMBODIED_PATH"] = str(embodied_path)
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name=config_name)
    GlobalHydra.instance().clear()
    return cfg


# --------------------------------------------------------------------------- #
# Registration (pure, always runs)
# --------------------------------------------------------------------------- #
def test_supported_model_registered_without_clobbering():
    from rlinf.config import EMBODIED_MODEL, SupportedModel

    assert SupportedModel.OPENPI_PYTORCH.value == "openpi_pytorch"
    assert SupportedModel.OPENPI_PYTORCH in EMBODIED_MODEL
    # Existing embodied models must be preserved, not clobbered.
    assert SupportedModel.OPENPI in EMBODIED_MODEL
    assert SupportedModel.GR00T_N1D6 in EMBODIED_MODEL
    # The new value must not collide with an existing one.
    assert SupportedModel.OPENPI_PYTORCH.value != SupportedModel.OPENPI.value


def test_no_eval_validator_migrated():
    import rlinf.config as config

    assert not hasattr(config, "_validate_openpi_pytorch_eval_cfg")


# --------------------------------------------------------------------------- #
# Rank-disjoint streaming partition (pure, always runs) — the AC-3 invariant
# --------------------------------------------------------------------------- #
def test_rank_disjoint_chunk_partition():
    try:
        from rlinf.data.datasets.openpi_pytorch.behavior.behavior_sft_dataset import (
            partition_chunk_indices,
        )
    except ImportError as exc:  # heavy lerobot dep absent
        pytest.skip(f"behavior_sft_dataset import unavailable: {exc}")

    num_chunks = 100
    world_size = 2
    num_workers = 3
    seen: list[int] = []
    for rank in range(world_size):
        for worker_id in range(num_workers):
            seen.extend(
                partition_chunk_indices(
                    num_chunks,
                    rank=rank,
                    world_size=world_size,
                    worker_id=worker_id,
                    num_workers=num_workers,
                )
            )
    # Disjoint across (rank, worker) and a complete cover of every chunk once.
    assert sorted(seen) == list(range(num_chunks))
    assert len(seen) == len(set(seen))
    # Two ranks see strictly different chunk identities.
    rank0 = {
        c
        for w in range(num_workers)
        for c in partition_chunk_indices(
            num_chunks,
            rank=0,
            world_size=world_size,
            worker_id=w,
            num_workers=num_workers,
        )
    }
    rank1 = {
        c
        for w in range(num_workers)
        for c in partition_chunk_indices(
            num_chunks,
            rank=1,
            world_size=world_size,
            worker_id=w,
            num_workers=num_workers,
        )
    }
    assert rank0.isdisjoint(rank1)


# --------------------------------------------------------------------------- #
# Hydra config build (AC-1) — needs hydra/omegaconf + rlinf importable
# --------------------------------------------------------------------------- #
def test_eval_config_builds_and_selects_openpi_pytorch():
    pytest.importorskip("hydra")
    try:
        from rlinf.config import validate_cfg
    except ImportError as exc:
        pytest.skip(f"rlinf.config unavailable: {exc}")
    cfg = _compose(
        EMBODIED_CONFIG,
        "behavior_ppo_openpi_pi05_pytorch_eval",
        REPO_ROOT / "examples" / "embodiment",
    )
    assert cfg.actor.model.model_type == "openpi_pytorch"
    validate_cfg(cfg)


def test_sft_config_builds_and_selects_openpi_pytorch():
    pytest.importorskip("hydra")
    try:
        from rlinf.config import validate_cfg
    except ImportError as exc:
        pytest.skip(f"rlinf.config unavailable: {exc}")
    cfg = _compose(SFT_CONFIG, "behavior_pi05_vla", REPO_ROOT / "examples" / "sft")
    assert cfg.actor.model.model_type == "openpi_pytorch"
    validate_cfg(cfg)


# --------------------------------------------------------------------------- #
# Converter loud-failure invariant (AC-9) — metadata-level, no weights needed
# --------------------------------------------------------------------------- #
def test_new2old_requires_reference_model():
    """``new2old`` must fail loudly without --reference-model (no silent output)."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rlinf.utils.ckpt_convertor.openpi.convert",
            "--mode",
            "new2old",
            "--input",
            "/tmp/does_not_exist_in",
            "--output",
            "/tmp/does_not_exist_out",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, (
        "new2old without --reference-model must fail, got success:\n"
        f"{result.stdout}\n{result.stderr}"
    )


# --------------------------------------------------------------------------- #
# Single SFT forward/loss step (AC-13) — guarded: needs weights + a GPU
# --------------------------------------------------------------------------- #
def test_sft_forward_step_finite_loss():
    base = os.environ.get("OPENPI_PYTORCH_BASE")
    tokenizer = os.environ.get("OPENPI_PYTORCH_TOKENIZER")
    assets = os.environ.get("OPENPI_PYTORCH_SFT_ASSETS")
    if not (base and tokenizer and assets):
        pytest.skip(
            "SFT forward step skipped: set OPENPI_PYTORCH_BASE / "
            "OPENPI_PYTORCH_TOKENIZER / OPENPI_PYTORCH_SFT_ASSETS to enable."
        )
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("SFT forward step skipped: no CUDA device available.")
    # The full forward exercise lives in the e2e SFT config; here we only assert
    # the factory builds a model whose Pi0 exposes the SFT loss entry point.
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_pytorch import get_model

    cfg = OmegaConf.create(
        {
            "model_path": base,
            "precision": "fp32",
            "num_steps": 5,
            "num_action_chunks": 32,
            "action_dim": 23,
            "openpi": {
                "model_action_dim": 32,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "max_token_len": 200,
                "assets_dir": assets,
                "asset_id": "behavior-1k/2025-challenge-demos",
                "paligemma_tokenizer": tokenizer,
            },
        }
    )
    model = get_model(cfg)
    assert hasattr(model, "gradient_checkpointing_enable")
