# North DFlash training scaffold

A bounded, CPU-only foundation for investigating a DFlash draft model matched to the local `North-Mini-Code-1.0-int4-autoround-gptq-g32` deployment.

**This is not a training-ready recipe.** It does not load North, download data, use a GPU, modify model files, or contact the running server. BF16 is mentioned only as a possible initializer; the target deployment remains the exact expert-only int4 AutoGPTQ model.

## Quick start

```bash
PYTHONPATH=src python3 -m north_dflash_training.cli dry-run
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The dry-run creates synthetic token sequences, samples bounded DFlash blocks, prints paper-aligned loss weights, derives a review-only North candidate from local JSON config, and compares disk/offline feature storage with online and ring-buffer estimates.

## Layout

- `src/north_dflash_training/schema.py` — response-example model and validation
- `src/north_dflash_training/sampling.py` — deterministic anchor/block sampler
- `src/north_dflash_training/weights.py` — exponential CE weights
- `src/north_dflash_training/cache.py` — feature-cache estimator
- `src/north_dflash_training/candidate.py` — config-only North derivation
- `src/north_dflash_training/cli.py` — synthetic CPU dry-run
- `schemas/response-example.schema.json` — interchange schema
- `configs/north-dflash-candidate.json` — generated review artifact
- `IMPLEMENTATION_STATUS.md` — implemented primitives and integration gaps
- `tests/` — dependency-free unit tests

## Scope and source of truth

Sampling and weighting follow the [DFlash paper](https://arxiv.org/abs/2602.06036), §4.2 and §A.1. The local reference inference checkout is `/home/douglasbrown/Code/dflash`; its architecture-specific model code is deliberately not copied into this foundation. North's local config/tokenizer are read-only inputs to candidate derivation.

See `IMPLEMENTATION_STATUS.md` before treating any future work as training or deployment.
