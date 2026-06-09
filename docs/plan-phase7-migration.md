# Plan — Phase 7: Migrate OpenPI-PyTorch BEHAVIOR pi0.5 SFT + Eval into Production RLinf

## Goal Description

Consolidate the new-PyTorch OpenPI (pi0.5 flow-matching VLA) **BEHAVIOR-task SFT training + evaluation** feature into the production repository and clean it up.

- **Source (read-only reference):** `/mnt/public/xzxuan/repos/RLinf_pi05`, branch `feat/openpi-pytorch-migration`. The feature is the net diff from base commit `6a8458731ac5219643a4501c101f0b99fded973f` to `HEAD` — 42 production files under `rlinf/` + `examples/` (~8.2k insertions), plus diagnostic `tests/unit_tests/` and `tools/` scaffolding that is explicitly **not** migrated.
- **Target (edit + test + review repo):** `/mnt/public/xzxuan/repos/RLinf`, branch `main`. All implementation, verification, and code review happen here. The target `main` has **diverged** from the feature's base commit (which is absent in the target), so every shared file the feature touches must be **3-way reconciled** against the current target — never patch-applied. The target already carries a *different* OpenPI integration (`rlinf/models/embodiment/openpi`, `openpi_cfg`), the GR00T N1.6 model, and a recent "avoid loading assets during config validation" BEHAVIOR fix; none of these may regress.
- **Correctness ground-truth (read-only):** `/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed` (config `pi05_b1k-task0000_sft_pytorch_mixed`).

The migration is a **port plus deliberate refactor** (relocate directories, merge/inline modules, delete diagnostic scaffolding, remove config knobs, consolidate checkpoint converters, push normalization/tokenization into well-placed homes, replace the `load_for_training` switch with a precision-driven dtype). It is **not** an attempt to reproduce byte-for-byte parity with the reference: per the user's explicit choices the production dataloader uses the **decentralized `per_rank_stream`** topology and evaluation runs with **non-deterministic (random) noise**. The acceptance bar is a *clean, working, non-regressing* feature: SFT loss descends, BEHAVIOR eval reaches roughly the expected success rate, all four checkpoint conversions work, and no existing target capability breaks.

**Scope decision (resolved): pi0.5 BEHAVIOR only.** The source branch's pi0.5 VLM-only SFT path (`_is_pi05_vlm_only`) and the Qwen2.5/Qwen3 VLM SFT configs are out of scope and are not migrated.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. Unless stated otherwise, all paths are in the **target** repo `/mnt/public/xzxuan/repos/RLinf`, and all tests run inside the project venv (`/mnt/public/xzxuan/.venv_pi`, Python 3.10).

- AC-1: The net-new `openpi_pytorch` model package is ported into the target and fully registered, and Hydra builds the config for **both** eval and SFT without breaking any existing target model path.
  - Positive Tests (expected to PASS):
    - `rlinf/models/embodiment/openpi_pytorch/` exists in the target with the pi0.5 model, action model, policy, and `pi0_model/` submodules.
    - `SupportedModel.OPENPI_PYTORCH` (string value `"openpi_pytorch"`) is registered in `rlinf/config.py` and added to `EMBODIED_MODEL`; `rlinf/models/__init__.py` registers a `_build_openpi_pytorch` builder reachable via `get_model`.
    - Building the eval config (`behavior_ppo_openpi_pi05_pytorch_eval`) and the SFT config (`behavior_pi05_vla`) through `build_config`/Hydra both succeed and select `model_type: openpi_pytorch`.
    - A config-build smoke for the *existing* target `openpi`, `openpi_cfg`, and `gr00t_n1d6` paths still succeeds.
  - Negative Tests (expected to FAIL):
    - Registering `openpi_pytorch` with a value that collides with an existing `SupportedModel` (e.g. `openpi`) is rejected.
    - Removing or renaming `GR00T_N1D6` / `OPENPI` from `SupportedModel` or `EMBODIED_MODEL` causes existing tests/config builds to fail (proves they were preserved, not clobbered).

- AC-2: Only the production feature surface is migrated; the diagnostic scaffolding is excluded, while the newly-requested docs and CI tests are added.
  - Positive Tests (expected to PASS):
    - No `docs/evidence/`, `tests/unit_tests/test_openpi_pytorch_*`, `tests/unit_tests/test_phase6_*`, `tools/sft_*`, `tools/_*`, or `tools/build_pinned_step0_npz.py` from the source feature appear in the target commit.
    - The source `.gitignore` and `CLAUDE.md` changes are **not** carried over.
    - The newly-authored docs (AC-11) and e2e CI yamls (AC-12) **are** present.
  - Negative Tests (expected to FAIL):
    - A grep of the target diff for `phase6`, `reference_fanout`, `pinned_inputs_npz`, `eval_deterministic_noise`, `id_only`, or `_step0_instrument` returns any match.

- AC-3: The BEHAVIOR SFT dataloader is `per_rank_stream`-only; all `reference_fanout` / pinned-loader / `id_only` machinery is removed; rank-disjoint partitioning is preserved; the dataset code is relocated and the transform is merged/inlined; and every loader parameter is read from YAML with no behavior-changing code defaults (except the fixed skill recipe, which stays hardcoded).
  - Positive Tests (expected to PASS):
    - The dataset package lives at `rlinf/data/datasets/openpi_pytorch/behavior/`; `behavior_sft_transform.py`'s contents are merged into `behavior_sft_data_loader.py`; `transform_behavior_sft_item` no longer exists as a standalone function (its body is inlined at the single use site).
    - `behavior_pinned_loader.py` is absent; `PinnedBehaviorSftDataLoader`, `build_pinned_behavior_sft_dataloader`, `resolve_loader_mode`, `builds_train_loader`, `reference_fanout_micro_batches`, the `REFERENCE_FANOUT` constant, and the `loader_mode`/`id_only` parameters are all removed across the dataset package and both SFT workers.
    - In a 2-rank, `num_workers>0` configuration, the per-`(rank, worker)` chunk partition (`partition_chunk_indices`, `global_worker_id = rank*num_workers + worker_id`, stride `world_size*num_workers`) yields **disjoint** `(episode_index, frame_index)` frame identities across ranks whose union covers every chunk exactly once.
    - The required YAML keys for `build_behavior_sft_dataloader` are enumerated in the example configs and read directly (`assets_dir`, `asset_id`, `micro_batch_size`, `eval_batch_size`, `use_skill`, `tasks`, `task_subtasks`, `behavior_dataset_root`, `repo_id`, `modalities`, `model_action_dim`, `num_action_chunks`, `max_token_len`, `num_workers`, `fine_grained_level`, `tolerance_s`, `seed`); the intentionally-fixed skill recipe (`skill_labels`, `enable_gap`, `allow_left`, `allow_right`) remains hardcoded with a one-line comment explaining why.
  - Negative Tests (expected to FAIL):
    - Omitting a required loader key from YAML raises a clear configuration error rather than silently falling back to a hidden default.
    - Any remaining import of `behavior_pinned_loader`, `REFERENCE_FANOUT`, or `id_only` causes an `ImportError`/`AttributeError`.

- AC-4: `behavior_sft_dataset.py` uses the OmniGibson utilities directly (`hf_transform_to_torch`, `aggregate_stats`, `decode_video_frames`, `VideoLoader`, `RGBVideoLoader`) instead of the ~250 lines of local reimplementation, and does so without triggering asset loading or heavyweight side effects at module import.
  - Positive Tests (expected to PASS):
    - The dataset module imports those symbols from `omnigibson.learning.utils.*`, and the previously-inlined reimplementations are deleted.
    - `python -c "import rlinf.data.datasets.openpi_pytorch.behavior.behavior_sft_dataset"` completes **without** loading OmniGibson scene/asset state (OmniGibson imports are lazy — performed inside dataset construction/use, not at module top level).
    - The dataset still produces correct rgb-modality frames for the pi0.5 SFT path.
  - Negative Tests (expected to FAIL):
    - A module-top-level OmniGibson import that loads assets at import time (regressing the target's "avoid loading assets during config validation" fix) is rejected.

- AC-5: The model factory replaces `load_for_training` with a precision-driven dtype, sources norm-stats and the tokenizer path from config, removes the digest/validation/selection helpers, and preserves the load-time invariants that keep SFT and eval correct.
  - Positive Tests (expected to PASS):
    - `load_for_training` is gone from `rlinf/models/embodiment/openpi_pytorch/__init__.py`; the model dtype is `torch_dtype_from_precision(cfg.precision)`; `examples/embodiment/config/model/pi0_5_pytorch.yaml` sets `precision: bf16` and `examples/sft/config/model/pi0_5_pytorch.yaml` sets `precision: fp32` (with `load_for_training` removed from both).
    - The precision guard that currently rejects non-bf16 (`__init__.py:175-185`) is extended/removed so that `precision: fp32` (→ `torch.float32`) is accepted; weight handling collapses to `model = model.to(torch_dtype)`.
    - Norm-stats load through a new `load_norm_stats(assets_dir, asset_id)` in `pi0_model/normalize.py` that returns stats from exactly `{assets_dir}/{asset_id}/norm_stats.json`; `resolve_norm_stats_dir`, `blank_asset_field`, `_is_blank`, `_select`, `_require_shape`, `_state_dict_metadata_digest`, `_file_digest`, `_validate_checkpoint_state_dict` are removed.
    - The tokenizer path is config-driven: an `actor.model.openpi.paligemma_tokenizer` field exists in both the eval and SFT configs and is passed as a `path` argument into `PaligemmaTokenizer.__init__`; no `_DEFAULT_EXTERNAL_TOKENIZER_PATH` hardcoded fallback remains in production code.
    - Load-time invariants hold: `model.load_state_dict(strict=True)` still enforces key/shape parity; loading a bf16 base checkpoint into the fp32 SFT model widens losslessly (the intended SFT init); the eval path loads a bf16 model; gradient checkpointing is active during SFT and inactive during eval (via the FSDP-manager path — see Implementation Notes); the action output remains `[B, 32, 23]` with the same fixed 23-of-32 env-dim slice.
  - Negative Tests (expected to FAIL):
    - A blank/whitespace `asset_id`, a missing `norm_stats.json`, or a wrong `assets_dir` raises a clear `FileNotFoundError` rather than silently resolving bare/empty stats.
    - A checkpoint with mismatched keys or shapes is rejected by `load_state_dict(strict=True)`.
    - Setting `precision: fp32` while the guard still hard-rejects non-bf16 raises a `ValueError` (proves the guard was actually extended).

- AC-6: `rlinf/workers/rollout/hf/huggingface_worker.py` removes deterministic-eval-noise machinery (random eval) while keeping the model-dispatch block and preserving existing target paths.
  - Positive Tests (expected to PASS):
    - `deterministic_eval_seed`, `_next_eval_noise_generator`, the `_eval_deterministic_noise`/`_eval_noise_seed`/`_eval_noise_step` attributes, and the `eval_rng` injection are removed; the `eval_deterministic_noise` and `eval_noise_seed` config keys are removed from all migrated configs.
    - The `if SupportedModel(self.cfg.actor.model.model_type) in [...]` dispatch block (the draft's "~line 404" block) is retained, with `SupportedModel.OPENPI_PYTORCH` added to the list **alongside** the existing `GR00T_N1D6`/`OPENPI` entries.
  - Negative Tests (expected to FAIL):
    - Naively overwriting the dispatch list so that `GR00T_N1D6` is dropped causes the GR00T eval path to fail.
    - Any residual reference to `eval_deterministic_noise`/`eval_noise_seed` in code or migrated YAML is flagged.

- AC-7: `rlinf/config.py` registers `openpi_pytorch` without migrating `_validate_openpi_pytorch_eval_cfg`, and preserves the target's GR00T N1.6 registration and the BEHAVIOR config-validation fix.
  - Positive Tests (expected to PASS):
    - `_validate_openpi_pytorch_eval_cfg` and its two call-sites are absent from the target.
    - `SupportedModel.OPENPI_PYTORCH` is added after `OPENPI` and included in `EMBODIED_MODEL`; `GR00T_N1D6` remains registered and in `EMBODIED_MODEL`.
    - `validate_embodied_cfg` retains the target's post-fix body (no OmniGibson asset load / yaml import at validation time).
  - Negative Tests (expected to FAIL):
    - A diff that reintroduces the OmniGibson-asset-loading block into `validate_embodied_cfg` is rejected.

- AC-8: The `warmup_optimizer_state` step-counter reset is migrated into `rlinf/utils/utils.py` with the comment condensed to two sentences, without disturbing the target-only rollout batch helpers.
  - Positive Tests (expected to PASS):
    - The ~17-line step-counter-reset block is appended inside `warmup_optimizer_state`; its explanatory comment is exactly two sentences.
    - The target-only helpers (`merge_rollout_epochs`, `preprocess_embodied_batch`, `flatten_embodied_batch`, `pack_batch`, `unpack_batch`) remain present and unchanged.
  - Negative Tests (expected to FAIL):
    - A migration that removes or alters any of the target-only rollout helpers is rejected.

- AC-9: The five checkpoint-conversion scripts are consolidated into a unified, de-duplicated tool under `rlinf/utils/ckpt_convertor/openpi/` covering the four modes, with a README, and any production runtime caller of the export API is updated.
  - Positive Tests (expected to PASS):
    - `rlinf/utils/ckpt_convertor/openpi/` exists with one shared core (prefix-strip, dtype-cast, safetensors/config IO, `copy_norm_stats` defined exactly once) and the four modes `jax2new`, `old2new`, `sft2new`, `new2old`, invoked via the target's `python -m rlinf.utils.ckpt_convertor.openpi.convert --mode {jax2new,old2new,sft2new,new2old} ...` argparse convention (or equivalent per-mode modules).
    - A `README.md` under that directory documents each mode's input/output checkpoint layout, dtype policy, norm-stats handling, and one example command per mode.
    - Any internal runtime caller of `export_sft_checkpoint_for_eval` / `export_sft_checkpoint_dir_for_eval` is repointed to the new location (no broken import); an import smoke for the production path passes.
    - Mode-specific invariants are preserved: `new2old` still requires `--reference-model` and fails loudly without it; `jax2new` still writes fp32 weights with a `dtype: bfloat16` config hint while `sft2new` casts floats to bf16; each mode copies/writes the correct `config.json` variant.
  - Negative Tests (expected to FAIL):
    - Running `new2old` without `--reference-model` does **not** silently emit an incomplete checkpoint.
    - A duplicated second definition of `copy_norm_stats` (the old `old_to_new.py` + `sft_to_new_pytorch.py` duplication) reappears.

- AC-10: Supporting modules are relocated to their requested homes and the dead `dataconfig` directory is removed.
  - Positive Tests (expected to PASS):
    - `normalize.py` and `tokenizer.py` live under `rlinf/models/embodiment/openpi_pytorch/utils/`; `processing.py`'s contents live under `rlinf/data/datasets/openpi_pytorch/behavior/` (reused with the SFT code where feasible, otherwise placed there intact); all importers are updated.
    - `rlinf/models/embodiment/openpi_pytorch/dataconfig/` is deleted.
  - Negative Tests (expected to FAIL):
    - Any stale import of the old `pi0_model/normalize.py`, `pi0_model/tokenizer.py`, `pi0_model/processing.py`, or `openpi_pytorch/dataconfig` paths raises `ImportError`.

- AC-11: English and Chinese documentation are added in the project style.
  - Positive Tests (expected to PASS):
    - A `sft_openpi_pytorch` doc is added under both `docs/source-en/rst_source/examples/embodied/` and `docs/source-zh/rst_source/examples/embodied/`, explaining how to run BEHAVIOR SFT with the new PyTorch OpenPI (key config + launch command).
    - `behavior.rst` (both `source-en` and `source-zh`) gains a section stating that eval is supported with the new PyTorch OpenPI code, in the style of the surrounding docs; both new docs are wired into the relevant toctree/index.
  - Negative Tests (expected to FAIL):
    - A doc that omits the key configuration or launch method, or that is added in only one language, is rejected.

- AC-12: Two e2e CI configs and a relocatable minimal CI fixture (`ci_behavior`) with README are added, reusing the established install path and dependency setup.
  - Positive Tests (expected to PASS):
    - `tests/e2e_tests/embodied/behavior_openpi_pytorch_eval.yaml` and `tests/e2e_tests/sft/behavior_openpi_pytorch_sft.yaml` exist and follow the `embodied-openpi-behavior-test` style of `.github/workflows/embodied-e2e-tests.yml` (create env, run script, clean up), installing via `bash requirements/install.sh embodied --model openpi --env behavior`.
    - A `/mnt/public/xzxuan/ci_behavior` directory is produced containing the model + a minimal `task-0000` subset, with a README describing the minimum requirements (model, tokenizer asset, norm-stats, dataset layout, env vars, launch commands) to run the CI.
    - **Install/deps (AC-12.1):** `requirements/install.sh embodied --model openpi --env behavior` installs everything `openpi_pytorch` SFT/eval/converters need (including `sentencepiece`, `safetensors`, and the tokenizer/model asset expectations); if a dependency is missing, it is added to the install path (and Docker stage if production uses Docker).
    - **Path hygiene (AC-12.2):** the migrated eval/SFT/CI configs contain no hardcoded machine-local `/mnt/...` asset paths baked into committed defaults; fixtures resolve from repo root or env vars so the CI is relocatable to `/workspace`.
  - Negative Tests (expected to FAIL):
    - A committed CI/example config that hardcodes a non-relocatable `/mnt/public/xzxuan/...` asset path is flagged.
    - A CI config that assumes the full `2025-challenge-demos` dataset (rather than the minimal `task-0000` subset) is rejected.

- AC-13: Automated correctness gates pass (the fast, non-GPU-heavy checks that replace the dropped diagnostic test suite).
  - Positive Tests (expected to PASS):
    - Import + config-build smoke for both eval and SFT configs.
    - A converter round-trip / metadata test per mode (key/shape/dtype of the output, mode-specific invariants from AC-9) — at minimum a dry/metadata-level check that does not require the full multi-GB weights when infeasible (and logs explicitly what was skipped).
    - A tiny dataloader test asserting rank-disjoint frame identities (AC-3) under a 2-rank/`num_workers>0` simulation.
    - A single SFT forward/loss step on a minimal batch produces a finite loss with a populated gradient (no NaN/Inf), at the smallest feasible scale.
  - Negative Tests (expected to FAIL):
    - Any of the above smoke checks raising an exception, or the dataloader test detecting cross-rank frame overlap, fails the gate.

- AC-14: Manual GPU acceptance gates confirm end-to-end behavior on real hardware/data.
  - Positive Tests (expected to PASS):
    - **SFT:** a real SFT run starts successfully and the training loss decreases (non-flat, downward trend) under the intended fp32-master / bf16-FSDP-compute setup.
    - **Eval:** running `behavior_ppo_openpi_pi05_pytorch_eval` with the converted SFT checkpoint `/mnt/public/xzxuan/repos/RLinf_pi05/logs/20260605-12:39:44-behavior_pi05_vla/pi05_sft_pytorch_new` over 128 trajectories yields **roughly 25–26%** success (soft trend target — "roughly that number is fine", not an exact threshold).
    - **Conversions:** all four conversions run end-to-end on the real artifacts — jax (`/mnt/public/xzxuan/models/pi05_base`), old (`/mnt/public/xzxuan/models/pi05_base_pytorch`), new (`/mnt/public/xzxuan/models/pi05_base_pytorch_new`), and the SFT-trained checkpoint (`/mnt/public/xzxuan/repos/RLinf_pi05/logs/20260605-12:39:44-behavior_pi05_vla/sft_behavior_pi05_vla`).
  - Negative Tests (expected to FAIL):
    - SFT loss stays flat or diverges (would indicate the fp32-master / FSDP-compute dtype contract was broken by the precision refactor).
    - Eval success is near 0% or wildly off ~25% (indicates a normalization/tokenizer/dtype/action-slice regression).

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices. The draft is highly prescriptive, so the bounds are narrow.

### Upper Bound (Maximum Acceptable Scope)
The full pi0.5 BEHAVIOR `openpi_pytorch` SFT + eval feature is added to the target repo exactly as the draft's nine sections specify: package ported and registered; dataloader cleaned to `per_rank_stream`-only and relocated under `rlinf/data/datasets/openpi_pytorch/behavior`; OmniGibson utilities used directly (lazily); model factory refactored to precision-driven dtype with config-driven norm-stats/tokenizer; converters consolidated under `rlinf/utils/ckpt_convertor/openpi` with a README; modules relocated and `dataconfig` removed; EN+ZH docs added; two e2e CI configs plus the `ci_behavior` fixture and README added; a thin automated smoke harness (AC-13) accompanies the change; and the manual GPU acceptance gates (AC-14) are demonstrated. All existing target capabilities (GR00T N1.6, the existing `openpi`/`openpi_cfg` integration, the BEHAVIOR validation fix, the StatefulDataLoader checkpointing) remain intact. The code is streamlined for clarity wherever the draft asks, with no over-engineering beyond the spec.

### Lower Bound (Minimum Acceptable Scope)
The `openpi_pytorch` pi0.5 BEHAVIOR feature is present and functional in the target: it builds via Hydra for eval and SFT; the dataloader is `per_rank_stream`-only with the fanout/pinned/id_only machinery removed and rank-disjointness preserved; the model factory precision refactor preserves the SFT-loss-descent and eval-correctness invariants (AC-5); the four converters work end-to-end (AC-14); EN+ZH docs and the two CI configs + `ci_behavior` README exist; and no existing target functionality regresses. Streamlining is "good enough for clarity" rather than maximal, and the automated harness covers at least import/config-build + the rank-disjoint dataloader check + one SFT forward step.

### Allowed Choices
- **Fixed by the draft (no discretion):** the exclusion list (no `docs/evidence`, `tests/unit_tests`, `tools/`, `.gitignore`, `CLAUDE.md`); `per_rank_stream`-only; the file relocations/merges/deletions; removal of `_validate_openpi_pytorch_eval_cfg`, `load_for_training`, the digest/select helpers, deterministic-eval-noise; the four converter modes; config-driven tokenizer path; the two example `precision` values (`bf16` eval / `fp32` SFT); EN+ZH docs; the two CI yaml paths; reuse of `install.sh embodied --model openpi --env behavior`.
- **Resolved decisions:** scope = pi0.5 BEHAVIOR only; gradient checkpointing via the existing FSDP-manager path (`fsdp_config.gradient_checkpointing`), not new factory plumbing; implementation + review happen in the target `RLinf` repo.
- **Can use (implementer's discretion):** the internal file granularity of the consolidated converter package (single dispatcher vs per-mode modules + shared core); the exact placement of the inlined transform and of the relocated `processing.py` contents (reused vs placed intact); the precise shape of the automated smoke harness; how `precision` guards are extended to admit fp32.
- **Cannot use:** patch-applying the source diff onto the target (must 3-way reconcile); reintroducing `reference_fanout`/pinned/`id_only`/deterministic-noise; module-top-level OmniGibson imports that load assets; hardcoded tokenizer/asset paths in committed production defaults; any change that drops GR00T N1.6, the existing `openpi`/`openpi_cfg` path, the BEHAVIOR validation fix, the rollout batch helpers, or the StatefulDataLoader checkpoint methods.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach
A two-pass **port-then-refactor** sequence (endorsed by both Codex passes) minimizes correctness drift:

1. **Pass A — port + reconcile (get it green):** Copy the net-new `openpi_pytorch` package, the dataset package, the example configs, and the converters into the target. Reconcile the seven shared files using the per-file map below — these are mostly additive insertions; the only structurally tricky one is `fsdp_vla_sft_worker.py` (start from the target's version and insert the `OPENPI_PYTORCH` branch, preserving its StatefulDataLoader checkpoint methods). Stand up the thin smoke harness (AC-13) and reach a buildable, importable state.
2. **Pass B — clean per the draft:** Apply the draft's deletions/merges/moves: strip `reference_fanout`/pinned/`id_only`; merge the transform and inline `transform_behavior_sft_item`; relocate the dataset package and the supporting modules; consolidate the converters; switch `load_for_training`→precision; swap to OmniGibson direct (lazy) imports; condense the `warmup_optimizer_state` comment; remove deterministic-eval-noise. Re-run the smoke harness after each move.
3. **Validate:** run the automated gates (AC-13), then the manual GPU gates (AC-14: SFT loss-descent, eval ~25-26%, four conversions). Author the EN+ZH docs and the CI configs + `ci_behavior` fixture/README.

Reconciliation cheat-sheet for the shared files (3-way against target `main`):
- `rlinf/models/__init__.py`, `rlinf/hybrid_engines/fsdp/utils.py`, `rlinf/utils/utils.py` — **additive, Low risk** (insert builder/registration; insert the `openpi_cosine`/`ref_warmup_cosine` lr branch + `import math`; append the warmup step-reset, leaving rollout helpers untouched).
- `rlinf/config.py`, `rlinf/workers/rollout/hf/huggingface_worker.py` — **Medium**: insert `OPENPI_PYTORCH` *alongside* `GR00T_N1D6` in the enum/dispatch list; drop the validator; keep the BEHAVIOR validation fix.
- `rlinf/workers/sft/fsdp_sft_worker.py` — **Medium**: per the draft, inline the dataloader init back into `__init__`, drop `_init_train_dataloader`/`_fanout_send`/`_fanout_recv`, and use the target's current `run_training` verbatim (none of the fanout-era modifications).
- `rlinf/workers/sft/fsdp_vla_sft_worker.py` — **High**: start from the target; insert the `OPENPI_PYTORCH` `build_dataloader` branch (calling `build_behavior_sft_dataloader`, no pinned dispatch) and a simplified `get_max_steps_per_epoch` branch (`len(self.data_loader) // self.gradient_accumulation`, no fanout); **preserve** the StatefulDataLoader `save_checkpoint`/`load_checkpoint`.
- `examples/` — copy the new BEHAVIOR eval/SFT yamls + `eval_embodiment.sh` Hydra-override forwarding; sanitize hardcoded paths; do not re-introduce the stale `value_lr`; keep `libero_spatial_ppo_gr00t_n1d6.yaml`.

### Relevant References
- `rlinf/data/datasets/behavior/` (source) — dataset package to relocate/clean; `partition_chunk_indices` is the rank-disjoint mechanism to preserve.
- `rlinf/models/embodiment/openpi_pytorch/__init__.py` (source) — the factory to refactor (lines ~163-291 region).
- `rlinf/models/embodiment/openpi_pytorch/pi0_model/{normalize,tokenizer,processing}.py` (source) — modules to relocate; add `load_norm_stats(assets_dir, asset_id)`.
- `rlinf/models/embodiment/openpi_pytorch/utils/{jax_to_new_pytorch,old_to_new,new_to_old,sft_to_new_pytorch,export_sft_checkpoint}.py` (source) — converters to consolidate.
- `/mnt/public/xzxuan/repos/RLinf/rlinf/utils/ckpt_convertor/` (target) — the existing convertor convention to match (`python -m ...`, argparse).
- `/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/src/behavior/learning/datas/dataset.py` (reference) — the OmniGibson-direct dataset pattern.
- `examples/sft/config/behavior_pi05_vla.yaml`, `examples/{sft,embodiment}/config/model/pi0_5_pytorch.yaml`, `examples/embodiment/config/behavior_ppo_openpi_pi05_pytorch_eval.yaml` (source) — configs to migrate, with the YAML key enumeration from AC-3.
- `.github/workflows/embodied-e2e-tests.yml` (target) — the `embodied-openpi-behavior-test` CI style to mirror.

## Dependencies and Sequence

### Milestones
1. **Reconcile & port (Pass A)**: land the feature in the target on the diverged baseline.
   - Phase A: Port the net-new packages (`openpi_pytorch` model, dataset package, converters) and example configs.
   - Phase B: 3-way reconcile the seven shared files using the cheat-sheet (preserve GR00T N1.6, `openpi`/`openpi_cfg`, BEHAVIOR fix, rollout helpers, StatefulDataLoader).
   - Phase C: Register `SupportedModel.OPENPI_PYTORCH` + builder + dispatch; reach importable/buildable state; stand up the AC-13 smoke harness.
2. **Refactor & clean (Pass B)**: apply the draft's deletions/merges/moves.
   - Step 1: Dataloader — `per_rank_stream`-only, remove fanout/pinned/id_only, merge transform + inline, relocate under `openpi_pytorch/behavior`, YAML-only params, OmniGibson direct (lazy).
   - Step 2: Model factory — precision refactor, `load_norm_stats`, config-driven tokenizer, drop digest/select/validate helpers, preserve invariants; condense `warmup_optimizer_state` comment; remove deterministic-eval-noise.
   - Step 3: Converters — consolidate under `ckpt_convertor/openpi` + README + update runtime export caller; relocate `normalize`/`tokenizer`/`processing`; delete `dataconfig`.
3. **Validate & document**: prove correctness and ship docs/CI.
   - Step 1: Automated gates (AC-13).
   - Step 2: Manual GPU gates (AC-14: SFT loss-descent, eval ~25-26%, four conversions).
   - Step 3: EN+ZH docs (AC-11); e2e CI yamls + `ci_behavior` fixture/README + install/deps + path hygiene (AC-12).

Dependencies: Milestone 1 precedes Milestone 2 (clean only what builds). Within Milestone 2, Step 1/Step 2 are largely independent but both precede the converter relocation in Step 3 (imports settle last). Milestone 3 depends on Milestone 2; AC-14 manual gates depend on AC-5/AC-3 being correct; the CI fixture (AC-12) depends on the configs being path-sanitized.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Port net-new `openpi_pytorch` model package, dataset package, converters, and example configs into target | AC-1, AC-2 | coding | - |
| task2 | 3-way reconcile the seven shared files against target `main` (preserve GR00T N1.6 / `openpi` / BEHAVIOR fix / rollout helpers / StatefulDataLoader) | AC-1, AC-6, AC-7, AC-8 | coding | task1 |
| task3 | Register `SupportedModel.OPENPI_PYTORCH` + `_build_openpi_pytorch` + SFT/eval dispatch; reach buildable state | AC-1 | coding | task2 |
| task4 | Stand up the automated smoke harness (import/config-build, rank-disjoint dataloader, single SFT forward step, converter metadata) | AC-13 | coding | task3 |
| task5 | Dataloader cleanup: `per_rank_stream`-only, remove fanout/pinned/id_only, merge transform + inline, relocate to `openpi_pytorch/behavior`, YAML-only params | AC-3 | coding | task3 |
| task6 | Switch `behavior_sft_dataset.py` to direct (lazy) OmniGibson utilities; streamline | AC-4 | coding | task5 |
| task7 | Model factory precision refactor: drop `load_for_training`, precision-driven dtype + fp32 guard, `load_norm_stats`, config-driven tokenizer, drop digest/select/validate helpers, preserve invariants | AC-5 | coding | task3 |
| task8 | Condense `warmup_optimizer_state` comment to two sentences; remove deterministic-eval-noise machinery | AC-6, AC-8 | coding | task2 |
| task9 | Consolidate converters under `rlinf/utils/ckpt_convertor/openpi/` (4 modes + shared core + README); update runtime export caller | AC-9 | coding | task1 |
| task10 | Relocate `normalize.py`/`tokenizer.py` to `openpi_pytorch/utils`; move `processing.py` into `data/datasets/openpi_pytorch/behavior`; delete `dataconfig/`; fix importers | AC-10 | coding | task5, task7 |
| task11 | Independent audit of the precision/dtype refactor + shared-file reconciliation (no regression, invariants preserved) | AC-5, AC-7 | analyze | task7, task2 |
| task12 | Author EN+ZH docs: `sft_openpi_pytorch` + `behavior.rst` eval section, wired into toctrees | AC-11 | coding | task3 |
| task13 | Author e2e CI yamls (eval + sft); build `ci_behavior` fixture + README; ensure install/deps + path hygiene | AC-12 | coding | task5, task7 |
| task14 | Run automated gates (AC-13) then manual GPU gates (AC-14): SFT loss-descent, eval ~25-26%, four conversions | AC-13, AC-14 | coding | task5, task6, task7, task9, task10 |

## Claude-Codex Deliberation

### Agreements
- Port-then-refactor (two passes) is the correct sequencing; a thin automated smoke harness must accompany the change to replace the dropped diagnostic suite.
- The shared files must be 3-way reconciled, not patch-applied; GR00T N1.6, the existing `openpi`/`openpi_cfg` integration, the BEHAVIOR config-validation fix, the rollout batch helpers, and the StatefulDataLoader checkpoint methods must all be preserved.
- `per_rank_stream` rank-disjointness (`partition_chunk_indices`) is independent of `reference_fanout` and survives its deletion.
- Dropping `_validate_checkpoint_state_dict` is acceptable because `load_state_dict(strict=True)` still enforces keys/shapes and bf16→fp32 widening is the intended (lossless) SFT init — provided the `precision` guard is extended to admit fp32 and the invariants in AC-5 are tested.
- Validation must be split into fast automated gates (AC-13) and manual GPU acceptance gates (AC-14); the eval ~25-26% figure is a soft trend target.

### Resolved Disagreements
- **Under-specification (Codex DISAGREE):** Codex judged candidate v1 under-specified. Resolution: adopted all REQUIRED_CHANGES — lazy-OG-import negative test (AC-4); a dedicated install/deps criterion (AC-12.1) and path-hygiene criterion (AC-12.2); a concrete norm-stats path contract and load-time invariants (AC-5); explicit handling of the `export_sft_checkpoint_for_eval` runtime caller (AC-9); SupportedModel uniqueness + existing-path smoke (AC-1); enumerated YAML keys with the fixed skill recipe carved out (AC-3); and the AC-13/AC-14 split.
- **Gradient checkpointing mechanism:** the draft's literal "call `gradient_checkpointing_enable()` if `runner.task_type == sft`" has a plumbing gap (the factory only receives `cfg.actor.model`). Resolution (user decision): use the existing FSDP-manager path (`fsdp_config.gradient_checkpointing`) — identical outcome, no new plumbing. AC-5 verifies the *outcome* (gradient checkpointing active for SFT, inactive for eval).
- **Cross-repo edit/review target:** Resolution (user decision): the target `RLinf` repo is the edit/test/review repo; `RLinf_pi05` is read-only reference. Any subsequent code review / RLCR loop should run from inside the target so it sees the real changes.

### Convergence Status
- Final Status: `converged`

## Pending User Decisions

_None. All decisions were resolved during planning:_

- DEC-1 (Scope): **Resolved — pi0.5 BEHAVIOR only.** The VLM-only path and Qwen SFT configs are out of scope.
- DEC-2 (Gradient checkpointing): **Resolved — FSDP-manager path** (`fsdp_config.gradient_checkpointing`), no factory plumbing.
- DEC-3 (Edit/review repo): **Resolved — implement & review in `/mnt/public/xzxuan/repos/RLinf`**; `RLinf_pi05` is read-only reference.

## Implementation Notes

### Cross-repo and review
- All edits land in `/mnt/public/xzxuan/repos/RLinf`. Treat `RLinf_pi05` as a read-only source of truth for the feature. Start any code review / RLCR loop from inside the target repo so the review compares against the diverged `main` and sees the real changes.

### Precision / dtype contract (do not break SFT loss descent)
- FSDP MixedPrecision `param_dtype: bf16` is the **compute** dtype and is configured independently in `behavior_pi05_vla.yaml`; it must stay bf16 and must **not** be coupled to `actor.model.precision`. With SFT `precision: fp32`, the model is handed to FSDP in fp32 (the master), FSDP casts to bf16 for forward/backward, and the optimizer updates the fp32 master — preserving the tiny warmup-LR AdamW updates. Verify SFT loss is non-flat (AC-14) as the guard against a dtype regression.
- Gradient checkpointing comes from the FSDP manager (`fsdp_config.gradient_checkpointing: True` for SFT, absent/False for eval); do not add factory-side `runner.task_type` plumbing.

### OmniGibson imports
- Import the OmniGibson utilities lazily inside dataset construction/use, never at module top level, to avoid loading assets at import (consistent with the target's recent BEHAVIOR validation fix).

### Converter export API
- Identify any production runtime caller of `export_sft_checkpoint_for_eval` / `export_sft_checkpoint_dir_for_eval` and repoint it to `rlinf/utils/ckpt_convertor/openpi/`. This is an internal repo, so a backward-compat shim for external importers is not required.

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead.

--- Original Design Draft Start ---

# Migration Task

I've written a lot of code to implement evaluation (eval) and SFT training for the new PyTorch OpenPI on the `behavior` task. The work you were involved in starts from commit `6a8458731ac5219643a4501c101f0b99fded973f` — everything modified or added between that commit and the current code was done to implement this feature.

What I need you to do now is to consolidate all the code involved here and implement this complete feature under `/mnt/public/xzxuan/repos/RLinf`. In other words, you must add all the code for this feature into the `/mnt/public/xzxuan/repos/RLinf` directory **and clean it up**.

Below are the specific code-style and migration requirements.

---

## 1. Content that is NOT needed — do NOT migrate

These are all intermediate artifacts and should not appear in the production code:

1. Everything under `docs/` (including `docs/`, `docs/evidence`)
2. `tests/unit_tests`
3. `tools/`
4. `.gitignore`, `CLAUDE.md`

## 2. `utils/utils`

The content under `utils/utils` should be migrated, but the comment block there is too long — condense it to two sentences.

## 3. `rlinf/workers/rollout/hf/huggingface_worker.py`

Anything related to fixing the eval noise and seed does NOT need to be migrated. I don't want them fixed — my future evals will just be random. Therefore you should also delete these two config options:

```yaml
eval_deterministic_noise: True
eval_noise_seed: 1234
```

However, the block around line 404 starting with `if SupportedModel(self.cfg.actor.model.model_type) ...` can stay.

## 4. Dataloader logic — use the decentralized approach

Our dataloader logic uses the decentralized approach. Therefore, remove the `data.loader_mode` config option — the migrated code should just use `per_rank_stream` mode. So, whether it's dataloader-related code or batch-fetching-related code, just delete the parts involving `reference_fanout`.

- In `rlinf/workers/sft/fsdp_sft_worker.py`, you added an `_init_train_dataloader` function — I want that reverted. Follow the original code and write this directly in `init`:

```python
if eval_only is not None or self.cfg.data.get("train_data_paths") is None:
    logging.warning(
        "Eval-only mode (eval_only=%s): skipping train dataloader",
        eval_only,
    )
    assert self.cfg.data.get("val_data_paths") is not None, (
        "val_data_paths must be set in eval-only mode"
    )
    self.data_loader = None
    self.data_iter = None
else:
    self.data_loader, self.data_config = self.build_dataloader(
        self.cfg.data.train_data_paths, eval_dataset=False
    )
    self.data_iter = iter(self.data_loader)
```

That's all — no need to add a separate function. Then, with minimal code, remove the `pinned_active` etc. logic. My understanding is that the original dataloader logic should perfectly implement `per_rank_stream` as is.

- Delete `_fanout_send` and `_fanout_recv`. The `run_training` logic should just use RLinf's current version — none of your modifications should be applied, since they were all for implementing fanout mode.

## 5. `rlinf/workers/sft/fsdp_vla_sft_worker.py`

- Line 45, the `if model_type == SupportedModel.OPENPI_PYTORCH:` block — keep this, but delete `build_pinned_behavior_sft_dataloader` (not needed). Consequently, when migrating, the entire file `rlinf/data/datasets/behavior/behavior_pinned_loader.py` should be dropped.
- For this block:

```python
if model_type == SupportedModel.OPENPI_PYTORCH:
    # reference_fanout: rank 0 consumes world_size*grad_accum micro-batches per
    # step (it pulls one rank's batches for every rank and scatters).
    per_step = self.gradient_accumulation * (
        self._world_size if getattr(self, "_reference_fanout", False) else 1
    )
    return max(1, len(self.data_loader) // per_step)
```

Verify the logic — if it's fanout mode, delete it as well.

## 6. `rlinf/config.py`

The modifications involving the `_validate_openpi_pytorch_eval_cfg` function should NOT be migrated. This `_validate_openpi_pytorch_eval_cfg` is a bit wasteful — it's enough to just write the default config file correctly.

## 7. Files under `rlinf/data/datasets/behavior`

- Delete `rlinf/data/datasets/behavior/behavior_pinned_loader.py`.
- Don't write a separate `transform_behavior_sft_item` function — just inline it where it's used.
- Move the contents of `rlinf/data/datasets/behavior/behavior_sft_transform.py` into `rlinf/data/datasets/behavior/behavior_sft_data_loader.py` — i.e., merge the two into one.
- Delete the `id_only` field; remove any code that referenced it, and remove it from the config too (not needed).
- `reference_fanout_micro_batches`, `builds_train_loader`, `resolve_loader_mode` — anything involving `REFERENCE_FANOUT` should also be deleted.
- In the `build_behavior_sft_dataloader` function in `rlinf/data/datasets/behavior/behavior_sft_data_loader.py`, delete `model_select`, `data_select`, and `data_cfg_select` — just select directly via `cfg.data.xxx`, with no defaults. All parameters here must be written out in the YAML, e.g. `assets_dir`, `asset_id`, `use_skill`, `tasks`. This includes the parameters passed to `create_behavior_sft_data_loader` — just grab them directly via `cfg.xxx`. All these parameters should be written in the YAML!
- In `rlinf/data/datasets/behavior/behavior_sft_dataset.py`, a lot of the code is rewritten by you and is unnecessary. Instead, like `/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/src/behavior/learning/datas/dataset.py`, just call the OmniGibson-related functionality directly. This includes `hf_transform_to_torch`, `aggregate_stats`, `decode_video_frames`, `VideoLoader`, `RGBVideoLoader`. Note that our environment has OmniGibson — just `import` it directly:

```
(.venv_pi) root@is-dcvh2anfn7ushuq7-devmachine-0:/mnt/public/xzxuan/repos/RLinf_pi05# uv pip list | grep omnigibson
Using Python 3.10.12 environment at: /mnt/public/xzxuan/.venv_pi
omnigibson                         3.7.2          /mnt/public/xzxuan/.venv_pi/BEHAVIOR-1K/OmniGibson
```

- The code under `rlinf/data/datasets/behavior` is a bit much and somewhat redundant. While strictly preserving correctness, appropriately streamline the code to make it clearer.
- Also, after migration, the contents under `rlinf/data/datasets/behavior` should be placed under `rlinf/data/datasets/openpi_pytorch/behavior`. This is because there may be other simulators in the future, so the code should be made more generic.

## 8. Contents under `rlinf/models/embodiment/openpi_pytorch`

- Delete the `rlinf/models/embodiment/openpi_pytorch/dataconfig` directory.
- For the several checkpoint-conversion scripts under `rlinf/models/embodiment/openpi_pytorch/utils` — can they be merged and streamlined? There are currently 5 related ones, but really there are only four modes: jax, old, new, SFT. The conversions implemented are: jax→new, old→new, SFT→new, new→old. See if the code can be streamlined and merged. Place all checkpoint-conversion code under `rlinf/utils/ckpt_convertor/openpi`, then add a README under it explaining how to use the conversion scripts.
- Move `rlinf/models/embodiment/openpi_pytorch/pi0_model/normalize.py` and `rlinf/models/embodiment/openpi_pytorch/pi0_model/tokenizer.py` into the `rlinf/models/embodiment/openpi_pytorch/utils` directory.
- Move the contents of `rlinf/models/embodiment/openpi_pytorch/pi0_model/processing.py` into `rlinf/data/datasets/openpi_pytorch/behavior`, and see whether some of it can be reused with the SFT code. Even if it can't be reused, still put it in that directory.
- `rlinf/models/embodiment/openpi_pytorch/__init__.py` is a key file to modify:
  - Delete `_state_dict_metadata_digest`, `_file_digest`, `_validate_checkpoint_state_dict`.
  - Delete `_select`, `_require_shape` — just grab values directly via `cfg.xxx`, no defaults. All values must be written out clearly in the YAML.
  - Delete lines 163–185; the default config is correct as is.
  - Lines 234–250 are too complex — just grab `norm_stats` directly from the config's `assets_dir` and `asset_id`. `resolve_norm_stats_dir`, `blank_asset_field`, `_is_blank` are all unnecessary. In `rlinf/models/embodiment/openpi_pytorch/pi0_model/normalize.py`, add a function `load_norm_stats` that takes `assets_dir` and `asset_id` and returns `norm_stats`.
  - There's a key parameter `load_for_training` — delete this parameter. Also note that `torch_dtype` is needed, which should be `torch_dtype_from_precision(cfg.precision)`. In `examples/embodiment/config/model/pi0_5_pytorch.yaml`, set `precision` to `bf16`; in `examples/sft/config/model/pi0_5_pytorch.yaml`, set `precision` to `fp32` (and also delete `load_for_training`). Then `expected_dtype = torch.float32 if load_for_training else torch.bfloat16` can be deleted.
  - Lines 263–277: just change to `model = model.to(torch_dtype)`.
  - Then, if `cfg.runner.task_type == sft`, call `model.gradient_checkpointing_enable()`.
  - Lastly, the current `rlinf/models/embodiment/openpi_pytorch/pi0_model/tokenizer.py` hardcodes `_DEFAULT_EXTERNAL_TOKENIZER_PATH`. Don't do this — add an `actor.model.openpi.paligemma_tokenizer` field in both the eval and train configs with this path, then pass `path` as a parameter into `__init__`.

## 9. Additional things to add

### 9.1 Docs

- Under `/mnt/public/xzxuan/repos/RLinf/docs/source-en/rst_source/examples/embodied`, add a `sft_openpi_pytorch` doc.
- Then in `/mnt/public/xzxuan/repos/RLinf/docs/source-en/rst_source/examples/embodied/behavior.rst`, add a section.

The former introduces how to perform SFT on the behavior task with the new PyTorch OpenPI; the latter adds a section to the doc stating that eval is supported with the new PyTorch OpenPI code. Write the docs following the style of the other docs — clearly explain the key configuration and the launch method. Note that both Chinese and English versions are needed: write them under `source-en` and `source-zh`.

### 9.2 CI tests

Write two CI tests:
- `/mnt/public/xzxuan/repos/RLinf/tests/e2e_tests/embodied/behavior_openpi_pytorch_eval.yaml`
- `/mnt/public/xzxuan/repos/RLinf/tests/e2e_tests/sft/behavior_openpi_pytorch_sft.yaml`

For now, use the data, environment, and model currently on this machine. But you must also give me a README describing the minimum requirements to run the CI you wrote — the core being the model and the dataset, especially the dataset. The full `/mnt/public/xzxuan/data/2025-challenge-demos` is too large; what I put under `/workspace` will only contain the `task-0000` data.

You should create a `ci_behavior` directory under `/mnt/public/xzxuan`, containing the model and the minimal data needed to run CI (a subset of the full `task-0000`). I'll copy it into `/workspace`. Also write a README explaining how to use it.

For the environment, both eval and SFT should follow the `embodied-openpi-behavior-test` style in `repos/RLinf/.github/workflows/embodied-e2e-tests.yml` — create the environment, run the script, clean up. Both SFT and eval install the same way: `bash requirements/install.sh embodied --model openpi --env behavior`.

---

## Validation

To verify the correctness of your implementation, you may want to do the following:

1. **Eval:** Run eval using the `behavior_ppo_openpi_pi05_pytorch_eval` config, with the model `/mnt/public/xzxuan/repos/RLinf_pi05/logs/20260605-12:39:44-behavior_pi05_vla/pi05_sft_pytorch_new`. It should reach roughly a 25–26% success rate (running 128 trajectories). It doesn't need to be exact — roughly that number is fine.

2. **SFT:** Be able to run SFT training successfully and observe the loss decreasing.

3. **Checkpoint inter-conversion:** Verify that all four conversions work:
   - jax (`/mnt/public/xzxuan/models/pi05_base`)
   - old (`/mnt/public/xzxuan/models/pi05_base_pytorch`)
   - new (`/mnt/public/xzxuan/models/pi05_base_pytorch_new`)
   - SFT-trained checkpoint (`/mnt/public/xzxuan/repos/RLinf_pi05/logs/20260605-12:39:44-behavior_pi05_vla/sft_behavior_pi05_vla`)


--- Original Design Draft End ---
