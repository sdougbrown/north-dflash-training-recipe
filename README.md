# North DFlash training scaffold

A bounded, CPU-only foundation for investigating target-matched DFlash draft models for local North AutoGPTQ INT4, Cohere NVFP4 W4A16, and Cohere FP8 verifiers.

**The local scaffold is not the training authority.** Deployable training uses the pinned upstream Speculators workflow in [docs/OFFICIAL_SPECULATORS_PIPELINE.md](docs/OFFICIAL_SPECULATORS_PIPELINE.md). The historical custom code remains for lifecycle tests and diagnosis. Each training branch must remain matched to one exact quantized verifier.

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

The random smoke artifact has passed a construction/load-only TP=2 gate against the exact AutoGPTQ target on a pinned ROCm 7.2.4 image. Subsequent bounded traces proved both the five-feature and eight-feature mappings for AutoGPTQ on Rocky, Cohere W4A16/MARLIN on Bitey, and Cohere FP8/TRITON on Bitey. Each produced ordered 2048-wide BF16 features with exact tokenizer alignment, but target feature tensors differ. Teacher extraction must disable prefix caching. The stock W4A16/MARLIN runtime exhibited small route-order nondeterminism; the pinned PR #48032 runtime now produces byte-identical repeated features. A fail-closed in-memory consumer validates and owns one connector result at a time with bounded queue backpressure. Separate live gates for AutoGPTQ TP=2, deterministic W4A16/MARLIN, and FP8/TRITON each cloned one request into that ring, stopped the exact teacher, and completed one draft-only optimizer step without writing a checkpoint. A follow-up FP8 gate then released the exact feature/lock handoff only after matching optimizer acknowledgement. The first retained FP8 micro-pilot subsequently completed eight updates on the eight-full-layer/five-feature candidate and round-tripped a draft-only checkpoint. Later serving diagnostics found that its original layout leaked the anchor's target state; those weights and their continuations are invalid for acceptance, while the storage/lifecycle proof remains valid. Training now uses target context strictly before the anchor. See [docs/RUNTIME_PROBE.md](docs/RUNTIME_PROBE.md) and [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

The dry-run creates synthetic token sequences, samples bounded DFlash blocks, builds and validates a CPU-only sparse layout relation, derives a review-only North candidate from local JSON config/tokenizer audit, and compares disk/offline feature storage with online and ring-buffer estimates.

For the optional CPU tensor adapter, install `.[torch]`. For validated
safetensors connector consumption, install `.[runtime]`. These modules are
isolated so importing the base package never imports PyTorch.

## Layout

- `src/north_dflash_training/schema.py` — response-example model and validation
- `src/north_dflash_training/sampling.py` — deterministic anchor/block sampler
- `src/north_dflash_training/layout.py` — dependency-free concatenated batch/layout relation
- `src/north_dflash_training/torch_layout.py` — optional PyTorch `[1, Q]` query tensors, dense `[Q, C + Q]` boolean visibility oracle, and FlexAttention block-mask construction
- `src/north_dflash_training/weights.py` — exponential CE weights
- `src/north_dflash_training/cache.py` — feature-cache estimator
- `src/north_dflash_training/feature_stream.py` — exact-runtime connector validation and fail-closed bounded in-memory feature ring; never deletes artifacts
- `src/north_dflash_training/connector_lifecycle.py` — orchestration-owned stable-file fingerprinting and post-optimizer transient release
- `src/north_dflash_training/save_resume.py` — immutable draft-only retained-pilot checkpoints with exact response/request ledgers, external manifest root, and trusted-local AdamW resume
- `src/north_dflash_training/candidate.py` — config/tokenizer-audited North derivation
- `src/north_dflash_training/teacher.py` — config-fingerprinted, checkpoint-unverified AutoGPTQ feature manifest, without extraction
- `src/north_dflash_training/checkpoint_identity.py` — explicit incremental config/index/shard hashing and verification without tensor loading
- `src/north_dflash_training/training.py` — optional-PyTorch typed teacher-feature, frozen shared-weight, adapter, and weighted-loss contract; includes a synthetic CPU-only adapter
- `src/north_dflash_training/transformers_draft_adapter.py` — optional, capability-gated eager-attention adapter for the real local `dflash.DFlashDraftModel`; it converts the audited bool `[Q, C + Q]` rule to a Qwen3 eager additive `[B, 1, Q, C + Q]` mask and is intentionally no-cache/non-FlexAttention
- `src/north_dflash_training/cli.py` — synthetic CPU dry-run and guarded random runtime-probe entrypoints
- `src/north_dflash_training/runtime_probe.py` — North-shaped config, static vLLM loader contract, deterministic random-only HF artifact writer
- `scripts/run_fp8_acceptance_first_micro8.py` — exact FP8 eight-example retained micro-pilot driver with teacher/trainer state separation
- `scripts/run_fp8_continuation_chunk.py` — cumulative one-to-eight-example FP8 continuation with parent-root verification and checkpoint round-trip
- `scripts/export_retained_draft.py` — verified draft-only checkpoint to vLLM-loadable HF artifact conversion without tied target I/O
- `scripts/measure_dflash_acceptance.py` — exact held-out speculative counters, per-position acceptance, and sequential throughput evidence
- `schemas/response-example.schema.json` — interchange schema
- `schemas/teacher-feature-manifest.schema.json` — config-level teacher identity manifest
- `configs/north-dflash-candidate.json` — generated review artifact
- `configs/north-int4-teacher-checkpoint-identity.json` — verified config/index/seven-shard identity for the AutoGPTQ teacher
- `configs/north-w4a16-teacher-checkpoint-identity.json` — independently verified config/index/four-shard identity for Cohere's W4A16 teacher
- `configs/north-autoround-gptq-runtime.json` — pinned Rocky TP=2 AutoGPTQ runtime, TP-complete trace, and one-step gate
- `configs/north-w4a16-deterministic-marlin-runtime.json` — pinned PR #48032 Marlin runtime, byte-identical trace, and one-step gate
- `configs/north-fp8-teacher-checkpoint-identity.json` — independently verified config/index/seven-shard identity for Cohere's FP8 teacher
- `configs/north-fp8-triton-runtime.json` — pinned Bitey FP8/TRITON runtime, byte-identical trace, and one-step gate
- `configs/north-fp8-acceptance-first-micro8.json` — first retained eight-example FP8 micro-pilot contract; not a geometry selection
- `configs/north-fp8-speculators-official-v1.json` — pinned upstream Speculators configuration and first end-to-end acceptance evidence
- `configs/north-fp8-speculators-code-scaling-v1.json` — 500/1,000/2,000-row on-policy scaling and fixed held-out acceptance evidence
- `configs/north-shared-embedding-identity.json` — byte-identical three-target tied embedding/output identity and audited mask row
- `docs/RUNTIME_PROBE.md` — artifact constraints, runtime identity rules, passed Rocky smoke evidence, and stop criteria
- `docs/OFFICIAL_SPECULATORS_PIPELINE.md` — authoritative North training workflow, bounded Bitey sequencing, and acceptance gate
- `IMPLEMENTATION_STATUS.md` — implemented primitives and integration gaps
- `tests/` — dependency-free unit tests

## Scope and source of truth

Sampling and weighting follow the [DFlash paper](https://arxiv.org/abs/2602.06036), §4.2 and §A.1. The local reference inference checkout is `/home/douglasbrown/Code/dflash`; its architecture-specific model code is deliberately not copied into this foundation. North's local config/tokenizer are read-only inputs to candidate derivation.

The base layout relation remains dependency-free and CPU-testable. With the optional PyTorch extra, the adapter materializes a single packed batch's `[1, Q]` query tensors and a dense `[Q, C + Q]` oracle: frozen context keys `0..C-1` precede query keys, each query sees target context strictly before its clean absolute anchor, and query-query attention is bidirectional only within its sampled block. The optional training contract accepts detached selected clean states in declared layer order, concatenates them to `[B, C, L*H]`, hands frozen embedding/LM-head modules directly to a draft-only adapter, and computes a weighted CE mean over the unshifted masked-future labels. The optional local-reference adapter proves eager Qwen3/DFlash mask construction, draft-only gradients, and bounded optimization. CPU fixtures remain tiny; three separate GPU construction gates used North dimensions, exact target-matched features, and exact tied vocabulary weights for one newly initialized one-layer draft update per verifier family. It supports no cache or FlexAttention and is not the production North attention path. The same visibility rule can construct a FlexAttention `BlockMask` when the installed build exposes `create_block_mask`; this is construction parity only, not an attention-kernel or GPU validation.

`checkpoint_identity.py` is also dependency-free and intentionally absent from the dry-run path. Exact AutoGPTQ, W4A16, and FP8 checkpoint identities are retained under `configs/`. Isolated runtime gates prove one-layer DFlash construction plus exact five- and eight-feature extraction for all three targets. `feature_stream.py` validates a transient connector artifact against its exact runtime/token/layer contract, clones one bounded request into `TeacherFeatureBundle`, and applies item/token/byte backpressure without writing a corpus. All three checkpoints also share one byte-identical tied embedding/output tensor; the exact mask row, frozen full-vocabulary projection, and TP=2 target/draft object identity pass. The draft-only save/resume contract retains exact response/request ledgers, connector-release proof, model/optimizer state, runtime identity, and an external manifest digest. Its lifecycle boundary passes. The old custom-pipeline weights remain invalidated by inclusive-anchor leakage and reference/runtime mismatch. The pinned upstream Speculators path now trains, emits standardized checkpoints that vLLM loads directly, and passes both parity and held-out acceptance gates. Fresh exact-FP8 on-policy coding pilots improved monotonically from 500 to 2,000 rows; the fixed 100-prompt holdout reached 10.44% draft-token acceptance, mean emitted length 1.731, and nonzero acceptance at all seven positions. This is held-out acceptance, not response-quality evidence. Further scaling must preserve normal noise augmentation, the untouched holdout, bounded transient features, and draft-only retained checkpoints. See `IMPLEMENTATION_STATUS.md` before treating any future work as deployment-ready.
