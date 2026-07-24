# Implementation status

This repository now contains a **bounded, CPU-only scaffold** for a North DFlash training investigation. It is not a training recipe and must not be treated as deployment-ready.

## Implemented and tested

- `ResponseExample` dataclass and `schemas/response-example.schema.json` for tokenized, target-generated prompt/response examples.
- Deterministic local-RNG anchor sampling based on DFlash paper §4.2 and §A.1: at most a configured number of distinct full blocks; one clean anchor followed by `block_size - 1` masked futures; no partial tail blocks.
- Dependency-free CPU batch layout for concatenated sampled blocks: flattened labels/loss masks/block IDs, absolute clean anchor positions, exponential position weights, bidirectional within-block query visibility, and frozen target-context visibility through each clean anchor. This is an auditable relation for tests, not a tensor or FlexAttention/GPU mask.
- Exponential per-future-position CE weights, `exp(-(k-1)/gamma)`, with paper defaults 16→7, 10→5, and 8→4.
- Lower-bound feature-cache estimator comparing dataset-wide offline storage, online per-batch working set, and a bounded ring buffer. It materializes no features.
- North candidate derivation from local `config.json`, `tokenizer_config.json`, and `tokenizer.json`, including the reference layer-spread formula, target vocab size 262144, and an audited `<MASK_TOKEN>` ID derived consistently from the tokenizer's special-token table and vocabulary.
- Config-fingerprinted, checkpoint-unverified AutoGPTQ teacher manifest recording complete quantization identity, selected zero-based block IDs, and target layer count; extraction is intentionally not implemented.
- Synthetic CLI dry-run. It performs no model loading, dataset download, GPU operation, model-file write, or server interaction.
- Standard-library unit tests covering schema validation, deterministic sampling, weighting, cache arithmetic, candidate derivation, and the dry-run path.

## Explicitly missing

- Tensor integration and the actual sparse block attention / Flex Attention mask. The CPU layout relation is tested for inter-block isolation and target-context leakage, but is not yet a model-consumable mask.
- Exact hidden-state extraction from the **expert-only int4 AutoGPTQ** North teacher. The current config digest cannot establish checkpoint-shard identity; no dequantized/BF16 teacher is substituted.
- Draft forward pass, target-feature projection/KV injection, optimizer, checkpointing, and actual training.
- North embedding/LM-head handoff; the tokenizer mask choice is now audited as `<MASK_TOKEN>` ID 1, but tied-weight behavior remains untested.
- vLLM Cohere2Moe auxiliary-state plumbing and target/draft integration.
- Any quality, acceptance, throughput, or loss benchmark.

BF16 appears only as a possible initializer in the candidate notes. It is not an alternative deployment target. The final target must remain the exact local `North-Mini-Code-1.0-int4-autoround-gptq-g32` expert-only AutoGPTQ deployment.

## Review gates before target integration

1. Confirm Cohere2Moe hidden-state numbering (including whether an embedding entry offsets block outputs) and what the serving stack can expose without changing expert routing or quantization.
2. Verify checkpoint-shard digests before extraction, then test tied embeddings and LM-head behavior for the audited tokenizer-derived mask ID 1.
3. Review whether a five-layer dense draft is appropriate for North's MoE target; this is a paper-shaped candidate, not a decision.
4. Convert the tested CPU layout relation into a reviewed tensor/FlexAttention integration and specify feature alignment before implementing a model forward path.
5. Integrate only through a non-running test harness first; do not modify the model directory or the running server.
