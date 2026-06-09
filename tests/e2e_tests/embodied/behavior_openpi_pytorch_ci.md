# OpenPI-PyTorch BEHAVIOR CI fixture

The `openpi_pytorch` BEHAVIOR end-to-end tests run two configs:

- eval: [`tests/e2e_tests/embodied/behavior_openpi_pytorch_eval.yaml`](behavior_openpi_pytorch_eval.yaml)
- sft:  [`tests/e2e_tests/sft/behavior_openpi_pytorch_sft.yaml`](../sft/behavior_openpi_pytorch_sft.yaml)

Both install via `bash requirements/install.sh embodied --model openpi --env behavior`
(which provides `sentencepiece`, `safetensors`, and OmniGibson through BEHAVIOR-1K),
mirroring the `embodied-openpi-behavior-test` job style (create env → run → clean up).
The workflow jobs are `embodied-openpi-pytorch-behavior-eval-test`
(`.github/workflows/embodied-e2e-tests.yml`) and `sft-openpi-pytorch-behavior-test`
(`.github/workflows/sft-e2e-tests.yml`).

## Minimal fixture

The committed configs hardcode **no** machine-local `/mnt/...` paths; every asset
path resolves from an env var with a `/workspace/ci_behavior/...` default, so the
fixture is relocatable. Assemble the minimal fixture (the full
`2025-challenge-demos` dataset is too large — only a `task-0000` subset is used)
with the helper + README under `/mnt/public/xzxuan/ci_behavior/` and stage it at
`/workspace/ci_behavior`:

```
ci_behavior/
  model/            # new-format eval checkpoint + model/physical-intelligence/behavior/norm_stats.json
  base_model/       # new-format fp32 SFT base checkpoint
  assets/           # assets/behavior-1k/2025-challenge-demos/norm_stats.json
  paligemma_tokenizer/paligemma_tokenizer.model
  dataset/          # minimal BEHAVIOR task-0000 LeRobot subset
```

Env vars (the CI jobs export these; defaults already point at `/workspace/ci_behavior`):

| Var | Used by | Default |
|-----|---------|---------|
| `OPENPI_PYTORCH_MODEL_PATH` | eval | `/workspace/ci_behavior/model` |
| `OPENPI_PYTORCH_ASSETS_DIR` | eval | `/workspace/ci_behavior/model` |
| `OPENPI_PYTORCH_BASE` | sft | `/workspace/ci_behavior/base_model` |
| `OPENPI_PYTORCH_SFT_ASSETS` | sft | `/workspace/ci_behavior/assets` |
| `BEHAVIOR_DATASET_ROOT` | sft | `/workspace/ci_behavior/dataset` |
| `OPENPI_PYTORCH_TOKENIZER` | both | `/workspace/ci_behavior/paligemma_tokenizer/paligemma_tokenizer.model` |

See `/mnt/public/xzxuan/ci_behavior/README.md` for the full layout, the per-artifact
minimum-requirements table, and `make_subset.sh` (which assembles the subset from the
full machine-local artifacts).
