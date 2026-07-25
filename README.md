# North DFlash training scaffold

A bounded, CPU-only foundation for investigating target-matched DFlash draft models for local North AutoGPTQ INT4, Cohere NVFP4 W4A16, and Cohere FP8 verifiers.

**This is not a training-ready recipe.** Its code does not load North, download data, use a GPU, modify model files, or contact a running server. BF16 is mentioned only as a possible draft initializer; each training branch must remain matched to one exact quantized verifier.

## Quick start

```bash
PYTHONPATH=src python3 -m north_dflash_training.cli dry-run
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Guarded pre-GPU runtime probe

`runtime-probe-config` prints, and `runtime-probe-generate` can guardedly write,
a **random, non-production** Hugging Face `DFlashDraftModel` for proving only
vLLM target auxiliary-state plumbing/runtime loading. The default is a
North-shaped one-layer/block-size-two smoke; the 8-layer full candidate requires
an explicit acknowledgement. It cannot overwrite an output and fails closed if
the local vLLM loader contract is not provable. It never trains, evaluates
acceptance, or starts a GPU/server. See [docs/RUNTIME_PROBE.md](docs/RUNTIME_PROBE.md).

The random smoke artifact has passed a construction/load-only TP=2 gate against the exact AutoGPTQ target on a pinned ROCm 7.2.4 image. Subsequent bounded traces proved both the five-feature and eight-feature mappings for AutoGPTQ on Rocky, Cohere W4A16/MARLIN on Bitey, and Cohere FP8/TRITON on Bitey. Each produced ordered 2048-wide BF16 features with exact tokenizer alignment, but target feature tensors differ. Teacher extraction must disable prefix caching; W4A16/MARLIN also exhibits small bounded numerical nondeterminism across repeated forwards. No training was attempted, and the repository still lacks a bounded online consumer. See [docs/RUNTIME_PROBE.md](docs/RUNTIME_PROBE.md) and [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

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
- `src/north_dflash_training/checkpoint_identity.py` — explicit incremental config/index/shard hashing and verification without tensor loading
- `src/north_dflash_training/training.py` — optional-PyTorch typed teacher-feature, frozen shared-weight, adapter, and weighted-loss contract; includes a synthetic CPU-only adapter
- `src/north_dflash_training/transformers_draft_adapter.py` — optional, capability-gated eager-attention adapter for the real local `dflash.DFlashDraftModel`; it converts the audited bool `[Q, C + Q]` rule to a Qwen3 eager additive `[B, 1, Q, C + Q]` mask and is intentionally no-cache/non-FlexAttention
- `src/north_dflash_training/cli.py` — synthetic CPU dry-run and guarded random runtime-probe entrypoints
- `src/north_dflash_training/runtime_probe.py` — North-shaped config, static vLLM loader contract, deterministic random-only HF artifact writer
- `schemas/response-example.schema.json` — interchange schema
- `schemas/teacher-feature-manifest.schema.json` — config-level teacher identity manifest
- `configs/north-dflash-candidate.json` — generated review artifact
- `configs/north-int4-teacher-checkpoint-identity.json` — verified config/index/seven-shard identity for the AutoGPTQ teacher
- `configs/north-w4a16-teacher-checkpoint-identity.json` — independently verified config/index/four-shard identity for Cohere's W4A16 teacher
- `configs/north-fp8-teacher-checkpoint-identity.json` — independently verified config/index/seven-shard identity for Cohere's FP8 teacher
- `docs/RUNTIME_PROBE.md` — artifact constraints, runtime identity rules, passed Rocky smoke evidence, and stop criteria
- `IMPLEMENTATION_STATUS.md` — implemented primitives and integration gaps
- `tests/` — dependency-free unit tests

## Scope and source of truth

Sampling and weighting follow the [DFlash paper](https://arxiv.org/abs/2602.06036), §4.2 and §A.1. The local reference inference checkout is `/home/douglasbrown/Code/dflash`; its architecture-specific model code is deliberately not copied into this foundation. North's local config/tokenizer are read-only inputs to candidate derivation.

The base layout relation remains dependency-free and CPU-testable. With the optional PyTorch extra, the adapter materializes a single packed batch's `[1, Q]` query tensors and a dense `[Q, C + Q]` oracle: frozen context keys `0..C-1` precede query keys, each query sees context through its clean absolute anchor, and query-query attention is bidirectional only within its sampled block. The optional training contract accepts detached selected clean states in declared layer order, concatenates them to `[B, C, L*H]`, hands frozen embedding/LM-head modules directly to a draft-only adapter, and computes a weighted CE mean over the unshifted masked-future labels. The optional local-reference adapter additionally proves a tiny random Qwen3/DFlash eager CPU forward, exact mask construction, draft-only gradients, and bounded optimization; it is not a North forward. It supports no cache or FlexAttention. The same visibility rule can construct a FlexAttention `BlockMask` when the installed build exposes `create_block_mask`; this is construction parity only, not an attention-kernel or GPU validation.

`checkpoint_identity.py` is also dependency-free and intentionally absent from the dry-run path. Exact AutoGPTQ, W4A16, and FP8 checkpoint identities are retained under `configs/`. Isolated runtime gates prove one-layer DFlash construction and exact five-feature extraction for all three targets, but this repository does not yet consume their features online. Bounded connector consumption, mask/embedding/LM-head training handoff, the eight-feature extraction gate, and actual training remain missing. See `IMPLEMENTATION_STATUS.md` before treating any future work as training or deployment.
