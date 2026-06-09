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

"""Export an SFT-trained checkpoint into the Phase-1 eval directory format.

An SFT run saves the :class:`OpenPiPytorchActionModel` wrapper state dict (keys
prefixed ``model.`` because the vendored ``Pi0`` lives at ``wrapper.model``).
The eval ``get_model`` factory instead loads a bare ``Pi0`` from a *new-format*
checkpoint directory containing ``model.safetensors`` (bf16, bare ``Pi0`` keys),
``config.json``, and the ``physical-intelligence/behavior/norm_stats.json``
asset tree. This module bridges the two so a model trained here can be evaluated
through the unchanged Phase-1 eval path.
"""

from __future__ import annotations

import json
import logging
import pathlib
import shutil
from typing import Any, Mapping

import torch

logger = logging.getLogger(__name__)

# Wrapper / FSDP prefixes that may sit in front of the bare ``Pi0`` keys in a
# saved checkpoint. ``OpenPiPytorchActionModel`` adds ``model.`` (the vendored
# Pi0 lives at ``wrapper.model``); FSDP/compile may add the others. No bare Pi0
# key begins with any of these, so stripping them is safe.
_WRAPPER_PREFIXES = (
    "_fsdp_wrapped_module.",
    "_orig_mod.",
    "module.",
    "model.",
)
_NORM_STATS_SUBDIR = pathlib.Path("physical-intelligence") / "behavior"


def _strip_wrapper_prefix(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Drop wrapper/FSDP key prefixes and cast float tensors to bf16.

    Removes any leading combination of the known wrapper/FSDP prefixes from each
    key so the result has bare ``Pi0`` keys. The eval loader validates that every
    checkpoint tensor is bf16, so float parameters are cast; integer/bool buffers
    (if any) are passed through.
    """
    bare: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        bare_key = key
        while True:
            for prefix in _WRAPPER_PREFIXES:
                if bare_key.startswith(prefix):
                    bare_key = bare_key[len(prefix) :]
                    break
            else:
                break
        if tensor.is_floating_point():
            tensor = tensor.to(torch.bfloat16)
        # Two distinct source keys must never collapse to the same bare key, or a
        # tensor would be silently dropped; fail loudly instead.
        if bare_key in bare:
            raise ValueError(
                f"duplicate bare key {bare_key!r} after prefix strip "
                "(two checkpoint keys collapsed to one); refusing to drop a tensor."
            )
        bare[bare_key] = tensor.detach().cpu().contiguous()
    return bare


def export_sft_checkpoint_for_eval(
    state_dict: Mapping[str, torch.Tensor],
    output_dir: str | pathlib.Path,
    *,
    config_json: Mapping[str, Any],
    norm_stats_dir: str | pathlib.Path | None = None,
    norm_stats: Mapping[str, Any] | None = None,
) -> pathlib.Path:
    """Write an SFT wrapper state dict to a Phase-1 eval checkpoint directory.

    Args:
        state_dict: The :class:`OpenPiPytorchActionModel` wrapper state dict
            (or an already-bare ``Pi0`` state dict).
        output_dir: Destination directory for the eval-format checkpoint.
        config_json: Model config written to ``config.json`` (e.g.
            ``action_horizon``, ``action_dim``, ``paligemma_variant``,
            ``action_expert_variant``).
        norm_stats_dir: Directory containing ``norm_stats.json`` to copy into the
            ``physical-intelligence/behavior`` asset tree.
        norm_stats: Inline norm-stats mapping to write when ``norm_stats_dir`` is
            not given. Exactly one of ``norm_stats_dir`` / ``norm_stats`` is
            required.

    Returns:
        The output directory path.
    """
    import safetensors.torch

    if (norm_stats_dir is None) == (norm_stats is None):
        raise ValueError(
            "Provide exactly one of norm_stats_dir or norm_stats to export the "
            "BEHAVIOR norm-stats asset tree."
        )

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bare_state = _strip_wrapper_prefix(state_dict)
    safetensors.torch.save_file(bare_state, str(output_dir / "model.safetensors"))
    (output_dir / "config.json").write_text(json.dumps(dict(config_json), indent=2))

    stats_dir = output_dir / _NORM_STATS_SUBDIR
    stats_dir.mkdir(parents=True, exist_ok=True)
    if norm_stats_dir is not None:
        src = pathlib.Path(norm_stats_dir) / "norm_stats.json"
        if not src.is_file():
            raise FileNotFoundError(f"norm_stats.json not found at {src}.")
        shutil.copyfile(src, stats_dir / "norm_stats.json")
    else:
        (stats_dir / "norm_stats.json").write_text(json.dumps(dict(norm_stats)))

    logger.info(
        "Exported SFT checkpoint to eval format at %s (%d tensors).",
        output_dir,
        len(bare_state),
    )
    return output_dir


def _as_state_dict(loaded: Any) -> Mapping[str, torch.Tensor]:
    """Unwrap common checkpoint containers down to a key->tensor mapping."""
    obj = loaded
    for _ in range(4):
        if (
            isinstance(obj, Mapping)
            and obj
            and all(isinstance(v, torch.Tensor) for v in obj.values())
        ):
            return obj
        if isinstance(obj, Mapping):
            for wrapper_key in ("model", "state_dict", "module"):
                if wrapper_key in obj and isinstance(obj[wrapper_key], Mapping):
                    obj = obj[wrapper_key]
                    break
            else:
                break
        else:
            break
    if isinstance(obj, Mapping) and obj:
        return obj
    raise TypeError(
        f"Could not extract a state dict from the checkpoint (got {type(loaded)!r})."
    )


def export_sft_checkpoint_dir_for_eval(
    checkpoint_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    *,
    config_json: Mapping[str, Any],
    norm_stats_dir: str | pathlib.Path | None = None,
    norm_stats: Mapping[str, Any] | None = None,
    weights_subpath: str = "actor/model_state_dict/full_weights.pt",
) -> pathlib.Path:
    """Export a saved RLinf SFT checkpoint *directory* into the eval format.

    Loads the consolidated full weights from ``{checkpoint_dir}/{weights_subpath}``
    — the RLinf FSDP save layout, e.g.
    ``checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt`` — then
    writes the Phase-1 eval directory via :func:`export_sft_checkpoint_for_eval`.

    Args:
        checkpoint_dir: A saved checkpoint directory (the ``global_step_<N>`` dir,
            its ``actor`` subdir, or a dir directly containing ``full_weights.pt``).
        output_dir: Destination eval-format directory.
        config_json / norm_stats_dir / norm_stats: As in
            :func:`export_sft_checkpoint_for_eval`.
        weights_subpath: Relative path to the consolidated weights file.
    """
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    candidates = [
        checkpoint_dir / weights_subpath,
        checkpoint_dir / "model_state_dict" / "full_weights.pt",
        checkpoint_dir / "full_weights.pt",
    ]
    weights_path = next((c for c in candidates if c.is_file()), None)
    if weights_path is None:
        raise FileNotFoundError(
            f"No full_weights.pt found under {checkpoint_dir} "
            f"(looked at {[str(c) for c in candidates]})."
        )
    loaded = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    state_dict = _as_state_dict(loaded)
    logger.info("Loaded SFT checkpoint weights from %s", weights_path)
    return export_sft_checkpoint_for_eval(
        state_dict,
        output_dir,
        config_json=config_json,
        norm_stats_dir=norm_stats_dir,
        norm_stats=norm_stats,
    )
