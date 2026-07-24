# Implementation status

This repository now contains a **bounded, CPU-only scaffold** for a North DFlash training investigation. It is not a training recipe and must not be treated as deployment-ready.

## Implemented and tested

- `ResponseExample` dataclass and `schemas/response-example.schema.json` for tokenized, target-generated prompt/response examples.
- Deterministic local-RNG anchor sampling based on DFlash paper §4.2 and §A.1: at most a configured number of distinct full blocks; one clean anchor followed by `block_size - 1` masked futures; no partial tail blocks.
- Exponential per-future-position CE weights, `exp(-(k-1)/gamma)`, with paper defaults 16→7, 10→5, and 8→4.
- Lower-bound feature-cache estimator comparing dataset-wide offline storage, online per-batch working set, and a bounded ring buffer. It materializes no features.
- North candidate derivation from the local `config.json` and `tokenizer_config.json`, including the reference layer-spread formula and the exact int4 AutoGPTQ deployment constraint.
- Synthetic CLI dry-run. It performs no model loading, dataset download, GPU operation, model-file write, or server interaction.
- Standard-library unit tests covering schema validation, deterministic sampling, weighting, cache arithmetic, candidate derivation, and the dry-run path.

## Explicitly missing

- Sparse block attention / Flex Attention mask implementation and proof of inter-block isolation.
- Exact hidden-state extraction from the **expert-only int4 AutoGPTQ** North teacher. No dequantized/BF16 teacher is substituted.
- Draft forward pass, target-feature projection/KV injection, optimizer, checkpointing, and actual training.
- North tokenizer-approved mask-token decision and embedding/LM-head handoff.
- vLLM Cohere2Moe auxiliary-state plumbing and target/draft integration.
- Any quality, acceptance, throughput, or loss benchmark.

BF16 appears only as a possible initializer in the candidate notes. It is not an alternative deployment target. The final target must remain the exact local `North-Mini-Code-1.0-int4-autoround-gptq-g32` expert-only AutoGPTQ deployment.

## Review gates before target integration

1. Confirm Cohere2Moe hidden-state numbering and what the serving stack can expose without changing expert routing or quantization.
2. Select and reserve a tokenizer-valid mask ID, then test tied embeddings and LM-head behavior.
3. Review whether a five-layer dense draft is appropriate for North's MoE target; this is a paper-shaped candidate, not a decision.
4. Specify sparse attention and feature alignment before implementing a model forward path.
5. Integrate only through a non-running test harness first; do not modify the model directory or the running server.
