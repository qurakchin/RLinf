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

"""Convert an RLinf SFT-trained checkpoint to the NEW-format PyTorch layout.

An SFT run (``bash examples/sft/run_vla_sft.sh behavior_pi05_vla``) saves the
:class:`OpenPiPytorchActionModel` wrapper state dict, consolidated by the FSDP
worker into::

    checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt

Every tensor key is prefixed ``model.`` because the vendored ``Pi0`` lives at
``wrapper.model`` (FSDP/compile may add ``_fsdp_wrapped_module.`` / ``_orig_mod.``
/ ``module.`` in front of that). This converter strips those wrapper prefixes,
casts float weights to bf16, and writes a directory **identical in structure and
key namespace to** ``/mnt/public/xzxuan/models/pi05_base_pytorch_new`` — i.e.
``model.safetensors`` (bf16, bare ``Pi0`` keys) plus ``config.json`` — so the
trained model can be evaluated through the unchanged new-format eval path. The
norm-stats file is copied across verbatim.

Usage (four-parameter interface; the two norm-stats paths are copied across)::

    python -m rlinf.models.embodiment.openpi_pytorch.utils.sft_to_new_pytorch \
        --ckpt              /mnt/public/xzxuan/repos/RLinf_pi05/logs/20260605-12:39:44-behavior_pi05_vla/sft_behavior_pi05_vla/checkpoints/global_step_30000 \
        --input-norm-stats  /mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/outputs/assets/train/pi05_b1k-task0000_sft_pytorch_mixed/behavior-1k/2025-challenge-demos/norm_stats.json \
        --output-model      /mnt/public/xzxuan/repos/RLinf_pi05/logs/20260605-12:39:44-behavior_pi05_vla/pi05_sft_pytorch_new \
        --output-norm-stats /mnt/public/xzxuan/repos/RLinf_pi05/logs/20260605-12:39:44-behavior_pi05_vla/pi05_sft_pytorch_new/physical-intelligence/behavior/norm_stats.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import torch

from rlinf.models.embodiment.openpi_pytorch.utils.export_sft_checkpoint import (
    _as_state_dict,
    _strip_wrapper_prefix,
)

# The BEHAVIOR pi0.5 architecture config — identical to
# ``pi05_base_pytorch_new/config.json``; the SFT fine-tunes the same architecture,
# so the converted checkpoint shares this exact config (and bf16 weights).
_PI05_BEHAVIOR_CONFIG = {
    "action_dim": 32,
    "action_horizon": 32,
    "max_token_len": 200,
    "paligemma_variant": "gemma_2b",
    "action_expert_variant": "gemma_300m",
    "pi05": True,
    "pcd": False,
    "dtype": "bfloat16",
}

# Relative paths to the consolidated full-weights file, tried in order, so
# ``--ckpt`` may point at the ``global_step_<N>`` dir, its ``actor`` subdir, the
# ``model_state_dict`` dir, or the ``full_weights.pt`` file itself.
_WEIGHTS_CANDIDATES = (
    "actor/model_state_dict/full_weights.pt",
    "model_state_dict/full_weights.pt",
    "full_weights.pt",
)


def _resolve_full_weights(ckpt: pathlib.Path) -> pathlib.Path:
    """Find the consolidated ``full_weights.pt`` for a saved SFT checkpoint path."""
    if ckpt.is_file():
        return ckpt
    candidates = [ckpt / rel for rel in _WEIGHTS_CANDIDATES]
    weights = next((c for c in candidates if c.is_file()), None)
    if weights is None:
        raise FileNotFoundError(
            f"No full_weights.pt found under {ckpt} "
            f"(looked at {[str(c) for c in candidates]})."
        )
    return weights


def copy_norm_stats(src: str | pathlib.Path, dst: str | pathlib.Path) -> None:
    """Copy the norm-stats file from ``src`` to ``dst`` verbatim (straight copy)."""
    src = pathlib.Path(src)
    dst = pathlib.Path(dst)
    if not src.is_file():
        raise FileNotFoundError(f"input norm stats not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def convert_sft_to_new(
    ckpt: str | pathlib.Path,
    input_norm_stats: str | pathlib.Path,
    output_model: str | pathlib.Path,
    output_norm_stats: str | pathlib.Path,
) -> pathlib.Path:
    """Convert an SFT checkpoint to the new-format layout (four-parameter interface).

    Loads the consolidated ``full_weights.pt`` under ``ckpt``, strips the wrapper /
    FSDP key prefixes and casts float weights to bf16 to recover the bare ``Pi0``
    state dict, writes ``output_model/model.safetensors`` + ``output_model/config.json``
    (the ``pi05_base_pytorch_new`` structure), and copies ``input_norm_stats``
    verbatim to ``output_norm_stats``.
    """
    import safetensors.torch

    weights_path = _resolve_full_weights(pathlib.Path(ckpt))
    loaded = torch.load(
        str(weights_path), map_location="cpu", weights_only=False, mmap=True
    )
    state_dict = _as_state_dict(loaded)
    bare_state = _strip_wrapper_prefix(state_dict)

    output_model = pathlib.Path(output_model)
    output_model.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(bare_state, str(output_model / "model.safetensors"))
    (output_model / "config.json").write_text(
        json.dumps(_PI05_BEHAVIOR_CONFIG, indent=2) + "\n"
    )

    copy_norm_stats(input_norm_stats, output_norm_stats)
    print(
        f"Converted {weights_path} -> {output_model} "
        f"({len(bare_state)} bf16 tensors); norm stats -> {output_norm_stats}"
    )
    return output_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        required=True,
        help="saved SFT checkpoint (global_step_<N> dir, its actor/ subdir, the "
        "model_state_dict/ dir, or the full_weights.pt file)",
    )
    parser.add_argument(
        "--input-norm-stats", required=True, help="norm_stats.json to copy across"
    )
    parser.add_argument(
        "--output-model",
        required=True,
        help="output new-format checkpoint dir (config.json + model.safetensors)",
    )
    parser.add_argument(
        "--output-norm-stats", required=True, help="destination norm_stats.json path"
    )
    args = parser.parse_args()
    convert_sft_to_new(
        args.ckpt,
        args.input_norm_stats,
        args.output_model,
        args.output_norm_stats,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
