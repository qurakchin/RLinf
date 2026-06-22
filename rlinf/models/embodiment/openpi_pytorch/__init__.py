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

"""Self-contained PyTorch OpenPI 0.5 model package for embodied BEHAVIOR.

Layout:
  openpi_action_model.py  Abstract base class (eval predict + dispatch).
  sft_action_model.py     SFT subclass (flow-matching loss).
  rl_action_model.py      RL/PPO subclass (SDE chain sampler + value head).
  pi0_model/              Vendored model core (pi0, gemma, siglip, ...).
  utils/                  Normalization, tokenizer (SFT only), image, rl-sampler.
  policies/               BEHAVIOR input/output transforms.

The BEHAVIOR streaming SFT dataset / data loader lives under
``rlinf.data.datasets.openpi_pytorch.behavior``; checkpoint conversion lives
under ``rlinf.utils.ckpt_convertor.openpi``.
"""

from __future__ import annotations

import logging
import pathlib

from rlinf.config import torch_dtype_from_precision

logger = logging.getLogger(__name__)


def get_model(cfg, torch_dtype=None):
    """Build the BEHAVIOR pi05 model from a model config (factory entry).

    The concrete wrapper class is selected by ``cfg.openpi.task``:

    * ``task: sft`` → :class:`OpenPiPytorchSFTActionModel` (built by
      :func:`_build_sft_model`).
    * ``task: rl``  → :class:`OpenPiPytorchRLActionModel` (built by
      :func:`_build_rl_model`; uses the upstream openpi.transforms pipeline
      so ``openpi.paligemma_tokenizer`` is not read).
    """
    import safetensors.torch
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_pytorch.pi0_model import gemma as pi0_gemma
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config

    model_cfg = cfg.openpi

    target_dtype = (
        torch_dtype
        if torch_dtype is not None
        else torch_dtype_from_precision(cfg.precision)
    )

    model_path = pathlib.Path(cfg.model_path)
    weights_path = model_path / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"openpi_pytorch checkpoint not found: {weights_path}")

    pi0_config = Pi0Config(
        pi05=True,
        action_horizon=int(cfg.num_action_chunks),
        action_dim=int(model_cfg.model_action_dim),
        paligemma_variant=str(model_cfg.paligemma_variant),
        action_expert_variant=str(model_cfg.action_expert_variant),
        dtype="bfloat16",
        pcd=False,
    )
    model = pi0_config.create()
    state_dict = safetensors.torch.load_file(str(weights_path), device="cpu")
    model.load_state_dict(state_dict, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    if target_dtype is not None:
        model = model.to(target_dtype)

    num_steps = int(cfg.num_steps)
    action_chunk = int(cfg.num_action_chunks)
    action_env_dim = int(cfg.action_dim)

    task = OmegaConf.select(model_cfg, "task", default=None)
    if task is None:
        raise ValueError(
            "actor.model.openpi.task is required: set it to 'sft' or 'rl' "
            "to pick the concrete OpenPI PyTorch model variant."
        )
    task = str(task).lower()

    logger.info(
        "openpi_pytorch[%s]: loaded %s (%.2fB params) strict from %s precision=%s "
        "num_steps=%s",
        task,
        pi0_config,
        n_params / 1e9,
        weights_path,
        cfg.precision,
        num_steps,
    )

    if task == "sft":
        return _build_sft_model(
            model_cfg,
            model,
            pi0_config=pi0_config,
            num_steps=num_steps,
            action_chunk=action_chunk,
            action_env_dim=action_env_dim,
        )

    if task == "rl":
        paligemma_width = pi0_gemma.get_config(pi0_config.paligemma_variant).width
        return _build_rl_model(
            cfg,
            model_cfg,
            model,
            num_steps=num_steps,
            action_chunk=action_chunk,
            action_env_dim=action_env_dim,
            paligemma_width=paligemma_width,
        )

    raise ValueError(
        f"actor.model.openpi.task={task!r} is not supported; "
        "use 'sft' or 'rl'."
    )


def _build_sft_model(
    model_cfg,
    model,
    *,
    pi0_config,
    num_steps,
    action_chunk,
    action_env_dim,
):
    """Build the SFT variant: vendored EvalProcessor + PaligemmaTokenizer."""
    from omegaconf import OmegaConf

    from rlinf.data.datasets.openpi_pytorch import get_eval_processer
    from rlinf.models.embodiment.openpi_pytorch.sft_action_model import (
        OpenPiPytorchSFTActionModel,
    )
    from rlinf.models.embodiment.openpi_pytorch.utils.normalize import load_norm_stats
    from rlinf.models.embodiment.openpi_pytorch.utils.tokenizer import (
        PaligemmaTokenizer,
    )

    norm_stats = load_norm_stats(model_cfg.assets_dir, model_cfg.asset_id)
    tokenizer_path = OmegaConf.select(
        model_cfg, "paligemma_tokenizer", default=None
    )
    if tokenizer_path is None:
        # Match the openpi default cache location so SFT YAMLs can stay
        # tokenizer-path-free if openpi has previously populated the file.
        tokenizer_path = (
            pathlib.Path("~/.cache/openpi/big_vision/paligemma_tokenizer.model")
            .expanduser()
        )
    tokenizer = PaligemmaTokenizer(
        tokenizer_path, max_len=pi0_config.max_token_len
    )
    env_type = model_cfg.get("env", "behavior")
    processor = get_eval_processer(
        env_type,
        norm_stats,
        tokenizer,
        action_chunk=action_chunk,
        action_env_dim=action_env_dim,
        model_action_dim=pi0_config.action_dim,
    )

    return OpenPiPytorchSFTActionModel(
        model,
        processor,
        num_steps=num_steps,
        action_env_dim=action_env_dim,
    )


def _build_rl_model(
    cfg,
    model_cfg,
    model,
    *,
    num_steps,
    action_chunk,
    action_env_dim,
    paligemma_width,
):
    """Build the RL variant: openpi.transforms pipeline (auto-downloads tokenizer).

    Mirrors :func:`rlinf.models.embodiment.openpi.__init__.get_model`'s wiring:
    consume :func:`get_openpi_config` for the upstream TrainConfig, derive
    ``data_config`` + ``norm_stats`` from the checkpoint, build the input/output
    transform lists, and install them on the model via ``setup_wrappers``. The
    PaliGemma tokenizer is created (and if necessary downloaded) inside that
    pipeline — the YAML does not specify a tokenizer path.
    """
    import openpi.shared.download as download
    import openpi.transforms as transforms
    from omegaconf import OmegaConf
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi_pytorch.rl_action_model import (
        OpenPiPytorchRLActionModel,
        OpenPiPytorchRLConfig,
    )

    config_name = str(OmegaConf.select(model_cfg, "config_name", default=""))
    if not config_name:
        raise ValueError(
            "actor.model.openpi.config_name is required for task='rl' "
            "(it selects the upstream openpi TrainConfig, e.g. 'pi05_behavior')."
        )
    data_kwargs = OmegaConf.select(cfg, "openpi_data", default=None)
    if data_kwargs is not None:
        data_kwargs = OmegaConf.to_container(data_kwargs, resolve=True)

    train_config = get_openpi_config(
        config_name, model_path=str(cfg.model_path), data_kwargs=data_kwargs
    )
    upstream_model_config = train_config.model

    data_config = train_config.data.create(
        train_config.assets_dirs, upstream_model_config
    )
    checkpoint_dir = download.maybe_download(str(cfg.model_path))
    if data_config.asset_id is None:
        raise ValueError("data_config.asset_id is required to load norm_stats.")
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)

    input_transforms = [
        transforms.InjectDefaultPrompt(None),
        *data_config.data_transforms.inputs,
        transforms.Normalize(
            norm_stats, use_quantiles=data_config.use_quantile_norm
        ),
        *data_config.model_transforms.inputs,
    ]
    output_transforms_list = [
        *data_config.model_transforms.outputs,
        transforms.Unnormalize(
            norm_stats, use_quantiles=data_config.use_quantile_norm
        ),
        *data_config.data_transforms.outputs,
    ]

    rl_cfg = OpenPiPytorchRLConfig(
        add_value_head=bool(
            OmegaConf.select(cfg, "add_value_head", default=False)
        ),
        noise_method=str(
            OmegaConf.select(model_cfg, "noise_method", default="flow_ode")
        ),
        noise_level=float(
            OmegaConf.select(model_cfg, "noise_level", default=0.0)
        ),
        joint_logprob=bool(
            OmegaConf.select(model_cfg, "joint_logprob", default=False)
        ),
        ignore_last=bool(
            OmegaConf.select(model_cfg, "ignore_last", default=False)
        ),
        value_after_vlm=bool(
            OmegaConf.select(model_cfg, "value_after_vlm", default=False)
        ),
        value_vlm_mode=str(
            OmegaConf.select(model_cfg, "value_vlm_mode", default="mean_token")
        ),
        detach_critic_input=bool(
            OmegaConf.select(model_cfg, "detach_critic_input", default=False)
        ),
        train_expert_only=bool(
            OmegaConf.select(model_cfg, "train_expert_only", default=False)
        ),
        config_name=config_name,
    )

    rl_model = OpenPiPytorchRLActionModel(
        model,
        num_steps=num_steps,
        action_chunk=action_chunk,
        action_env_dim=action_env_dim,
        rl_cfg=rl_cfg,
        paligemma_width=paligemma_width,
    )
    rl_model.setup_wrappers(input_transforms, output_transforms_list)
    if bool(OmegaConf.select(model_cfg, "train_expert_only", default=False)):
        # Mirror the legacy ``openpi/openpi_action_model.OpenPi0ForRLActionPrediction``
        # PPO path: freeze the PaliGemma VLM (SigLIP vision + LLM expert 0) and
        # only update the action expert + projections + value head. With the VLM
        # frozen, the autograd graph dead-ends at the prefix output, so PPO
        # backward never traverses the 2.5B paligemma — this is the dominant
        # ``actor/run_training`` parity lever vs the legacy implementation.
        frozen = rl_model.freeze_vlm()
        logger.info(
            "openpi_pytorch[rl]: train_expert_only=True; froze %d parameter tensors "
            "(SigLIP + gemma expert-0)",
            frozen,
        )
    return rl_model
