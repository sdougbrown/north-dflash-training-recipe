# North DFlash training scaffold

A bounded, CPU-only foundation for investigating a DFlash draft model matched to the local `North-Mini-Code-1.0-int4-autoround-gptq-g32` deployment.

**This is not a training-ready recipe.** It does not load North, download data, use a GPU, modify model files, or contact the running server. BF16 is mentioned only as a possible initializer; the target deployment remains the exact expert-only int4 AutoGPTQ model.

## Quick start

```bash
PYTHONPATH=src python3 -m north_dflash_training.cli dry-run
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The dry-run creates synthetic token sequences, samples bounded DFlash blocks, builds and validates a CPU-only sparse layout relation, derives a review-only North candidate from local JSON config/tokenizer audit, and compares disk/offline feature storage with online and ring-buffer estimates.

For the optional CPU tensor adapter, install `.[torch]`. It lives in
`north_dflash_training.torch_layout`, so importing the base package never
imports PyTorch.

## Layout

- `src/north_dflash_training/schema.py` — response-example model and validation
- `src/north_dflash_training/sampling.py` — deterministic anchor/block sampler
- `src/north_dflash_training/layout.py` — dependency-free concatenated batch/layout relation
- `src/north_dflash_training/torch_layout.py` — optional PyTorch `[1, Q]` query tensors, dense `[Q, C + Q]` boolean visibility oracle, and FlexAttention block-mask construction
- `src/north_dflash_training/weights.py` — exponential CE weights
- `src/north_dflash_training/cache.py` — feature-cache estimator
- `src/north_dflash_training/candidate.py` — config/tokenizer-audited North derivation
- `src/north_dflash_training/teacher.py` — config-fingerprinted, checkpoint-unverified AutoGPTQ feature manifest, without extraction
- `src/north_dflash_training/cli.py` — synthetic CPU dry-run
- `schemas/response-example.schema.json` — interchange schema
- `schemas/teacher-feature-manifest.schema.json` — config-level teacher identity manifest
- `configs/north-dflash-candidate.json` — generated review artifact
- `IMPLEMENTATION_STATUS.md` — implemented primitives and integration gaps
- `tests/` — dependency-free unit tests

## Scope and source of truth

Sampling and weighting follow the [DFlash paper](https://arxiv.org/abs/2602.06036), §4.2 and §A.1. The local reference inference checkout is `/home/douglasbrown/Code/dflash`; its architecture-specific model code is deliberately not copied into this foundation. North's local config/tokenizer are read-only inputs to candidate derivation.

The base layout relation remains dependency-free and CPU-testable. With the optional PyTorch extra, the adapter materializes a single packed batch's `[1, Q]` query tensors and a dense `[Q, C + Q]` oracle: frozen context keys `0..C-1` precede query keys, each query sees context through its clean absolute anchor, and query-query attention is bidirectional only within its sampled block. The same rule can construct a FlexAttention `BlockMask` when the installed build exposes `create_block_mask`; this is construction parity only, not an attention-kernel or GPU validation. Teacher extraction and all model/training work remain missing. See `IMPLEMENTATION_STATUS.md` before treating any future work as training or deployment.
