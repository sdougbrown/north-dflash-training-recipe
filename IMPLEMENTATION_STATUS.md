# Implementation status

This repository now contains a **bounded, CPU-only scaffold** for a North DFlash training investigation. It is not a training recipe and must not be treated as deployment-ready.

## Implemented and tested

- `ResponseExample` dataclass and `schemas/response-example.schema.json` for tokenized, target-generated prompt/response examples.
- Deterministic local-RNG anchor sampling based on DFlash paper §4.2 and §A.1: at most a configured number of distinct full blocks; one clean anchor followed by `block_size - 1` masked futures; no partial tail blocks.
- Dependency-free CPU batch layout for concatenated sampled blocks: flattened labels/loss masks/block IDs, absolute clean anchor positions, exponential position weights, bidirectional within-block query visibility, and frozen target-context visibility through each clean anchor.
- Optional `.[torch]` tensor adapter, isolated in `north_dflash_training.torch_layout`: one packed CPU-neutral `[1, Q]` query batch with int64 IDs/labels/positions, bool loss mask, floating weights, plus a dense bool `[Q, C + Q]` oracle. Its first `C` keys represent frozen absolute target-context positions; every query sees only `0..absolute_anchor`, while query keys remain bidirectional within a block and isolated across blocks. The identical predicate can construct a FlexAttention `BlockMask` when an installed PyTorch exposes `create_block_mask`; CPU tests cover tensor values/dtypes/shapes, predicate/oracle parity, context boundary, and block isolation.
- Exponential per-future-position CE weights, `exp(-(k-1)/gamma)`, with paper defaults 16→7, 10→5, and 8→4.
- Lower-bound feature-cache estimator comparing dataset-wide offline storage, online per-batch working set, and a bounded ring buffer. It materializes no features.
- North candidate derivation from local `config.json`, `tokenizer_config.json`, and `tokenizer.json`, including the reference layer-spread formula, target vocab size 262144, and an audited `<MASK_TOKEN>` ID derived consistently from the tokenizer's special-token table and vocabulary.
- Config-fingerprinted, checkpoint-unverified AutoGPTQ teacher manifest recording complete quantization identity, selected zero-based block IDs, and target layer count; exact teacher extraction is intentionally not implemented.
- Dependency-free exact checkpoint identity tooling that incrementally hashes explicitly requested `config.json`, a Hugging Face weight index, and its declared shards without loading tensors. The installed North checkpoint was hashed and independently reverified in `configs/north-int4-teacher-checkpoint-identity.json`: 19,882,132,698 bytes across config, index, and seven shards; manifest SHA-256 `fca065d1f7df51b17763e5555363c36100d72c638cf318f71ef6c7f9b5319cae`.
- Optional `.[torch]` training-step contract in `north_dflash_training.training`: typed detached clean teacher-state bundles with declared layer ordering, concatenation to `[B, C, L*H]`, direct frozen shared embedding/LM-head references, an architecture-specific draft-adapter boundary, and a weighted CE mean over exactly non-ignored masked future labels. The tiny `SyntheticDraftAdapter` proves only CPU gradient flow/loss reduction; it is not a Qwen3 or North draft.
- A separately isolated, capability-gated `transformers_draft_adapter` skeleton documents the locally inspected `dflash.DFlashDraftModel` call boundary but blocks forward execution until target/KV positional and mask semantics are reviewed. Base imports do not require PyTorch or Transformers.
- Synthetic CLI dry-run. It performs no model loading, dataset download, GPU operation, model-file write, shard hash, or server interaction.
- Standard-library unit tests covering schema validation, deterministic sampling, weighting, cache arithmetic, candidate derivation, and the dry-run path.

## Explicitly missing

- A model-specific attention integration. The optional adapter supplies tensor inputs, a dense oracle, and FlexAttention block-mask construction parity, but does not build query/KV states, inject frozen teacher features, invoke an attention kernel, or validate CPU/GPU kernel behavior.
- Exact hidden-state extraction from the **expert-only int4 AutoGPTQ** North teacher. Checkpoint file identity is now established, but no hidden states have been extracted and no dequantized/BF16 teacher is substituted.
- A North/model-specific draft forward pass, target-feature projection/KV injection, attention-kernel integration, optimizer policy, checkpointing, and actual training. The optional CPU contract is a bounded interface test, not a target training recipe.
- North embedding/LM-head handoff. The contract freezes direct module references and tests draft-only gradients with synthetic modules, but North's tied-weight behavior and exact module paths remain untested.
- Runtime validation and target/draft integration of the isolated vLLM Cohere2Moe auxiliary-state branch (`experiment/north-dflash`, commit `a1cad4f67`). Static checks pass, but it has not been built or run against the real TP=2 target.
- Any quality, acceptance, throughput, or loss benchmark.

BF16 appears only as a possible initializer in the candidate notes. It is not an alternative deployment target. The final target must remain the exact local `North-Mini-Code-1.0-int4-autoround-gptq-g32` expert-only AutoGPTQ deployment.

## Review gates before target integration

1. Confirm Cohere2Moe hidden-state numbering (including whether an embedding entry offsets block outputs) and what the serving stack can expose without changing expert routing or quantization.
2. Use the retained exact checkpoint identity manifest as an extraction preflight, then test tied embeddings and LM-head behavior for the audited tokenizer-derived mask ID 1.
3. Review whether a five-layer dense draft is appropriate for North's MoE target; this is a paper-shaped candidate, not a decision.
4. Specify target-feature/KV alignment and review model-specific use of the optional tensor/FlexAttention visibility adapter before enabling a model forward path; the Transformers skeleton intentionally raises instead of guessing this conversion, and predicate construction is not kernel validation.
5. Integrate only through a non-running test harness first; do not modify the model directory or the running server.
