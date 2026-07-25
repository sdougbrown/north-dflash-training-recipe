# Three-target North DFlash execution plan

When the deployed verifier changes checkpoint or precision, its generated tokens and hidden states can change too. A drafter that accepts well against one quantized North verifier is not automatically matched to another. This plan produces and evaluates separate DFlash families for AutoRound GPTQ INT4, Cohere's QAD-trained NVFP4 W4A16 release, and Cohere's FP8 release. The filename is retained to avoid breaking existing references.

**Reader:** the operator planning North-specific training and serving validation. **Outcome:** three independently evidenced, target-matched BF16 draft checkpoints, with an optional FP8-weight export evaluated for each. **Prerequisites:** the candidate geometry review in [North draft candidates](NORTH_DRAFT_CANDIDATES.md), the current architectural boundaries in [Architecture decisions](ARCHITECTURE_DECISIONS.md), and the known implementation gaps in [Implementation status](../IMPLEMENTATION_STATUS.md).

The primary objective is acceptance against the exact production target distribution. A BF16 North model, a dequantized copy, or a differently quantized checkpoint is not a substitute for any deployed verifier.

## Two independent precision axes

Treat this as a pairing problem, like fitting a key to one lock at a time:

- **Verifier target** identifies the exact North checkpoint and serving behavior that generates responses, produces hidden states, and verifies draft tokens. It is `NorthINT4Target` (expert-only AutoGPTQ group-32), `NorthW4A16Target` (Cohere's expert-only NVFP4 group-16/QAD checkpoint), or `NorthFP8Target` (Cohere's FP8 checkpoint).
- **Draft-weight precision** identifies only how a trained draft is stored and served: canonical `BF16Draft`, or optional post-training `FP8Draft`.

Every branch trains its canonical artifact in BF16. FP8 is not a teacher choice and is not a second training target; it is a post-training deployment export that must be measured against the same target and must win empirically before use.

Do not pool AutoGPTQ, W4A16, and FP8 features into one unconditioned draft and call it target matched. Shared prompts, candidate definitions, base/random initialization hashes, code, and infrastructure are allowed. Target-generated response tokens, online features, final adaptation state, checkpoint identity, evaluation evidence, and promotion decisions are not shared.

A branch may warm-start from the other target only as a declared experiment. Its record must identify the source checkpoint and its target, then repeat final adaptation, matched evaluation, and all acceptance gates against the destination target. Warm-start evidence does not transfer promotion eligibility.

## Artifact matrix and names

The following are the only promotion candidates. The BF16 rows are canonical trained artifacts. FP8 rows are optional exports, not required deliverables.

| Verifier target | Draft-weight precision | Artifact name | Role | Promotion eligibility |
| --- | --- | --- | --- | --- |
| exact INT4 AutoGPTQ | BF16 | `North-DFlash-NorthINT4Target-BF16Draft` | canonical INT4-paired training checkpoint | yes, with the INT4 verifier only |
| exact INT4 AutoGPTQ | FP8 | `North-DFlash-NorthINT4Target-FP8Draft` | post-training export of the INT4 BF16 draft | only if it wins the INT4 BF16-versus-FP8 comparison |
| exact Cohere NVFP4 W4A16 | BF16 | `North-DFlash-NorthW4A16Target-BF16Draft` | canonical W4A16-paired training checkpoint | yes, with the W4A16 verifier only |
| exact Cohere NVFP4 W4A16 | FP8 | `North-DFlash-NorthW4A16Target-FP8Draft` | post-training export of the W4A16 BF16 draft | only if it wins the W4A16 BF16-versus-FP8 comparison |
| exact FP8 | BF16 | `North-DFlash-NorthFP8Target-BF16Draft` | canonical FP8-paired training checkpoint | yes, with the FP8 verifier only |
| exact FP8 | FP8 | `North-DFlash-NorthFP8Target-FP8Draft` | post-training export of the FP8 BF16 draft | only if it wins the FP8 BF16-versus-FP8 comparison |

Each artifact manifest must include its verifier-target label; source checkpoint identity manifest and hashes; tokenizer and prompt-set identities; target-layer IDs and verified extractor indexing; geometry; initialization/warm-start lineage; draft-weight precision; training/evaluation run IDs; serving revision; and the exact acceptance report. A name alone is not identity proof.

## Current evidence and boundaries

### Complete evidence

- The repository has CPU-only sampling, layout, feature-bundle, frozen-shared-weight, and weighted-loss contracts. The optional reference adapter establishes an eager, no-cache Qwen/DFlash forward, not a North forward.
- The retained INT4 checkpoint identity manifest hashes its config, weight index, and seven declared shards. The INT4 configuration identifies a 4-bit integer AutoGPTQ/AutoRound checkpoint with group size 32 and expert-only requirements.
- Both geometry candidates remain review candidates: 8 full draft layers with 5 target features, and 6 layers with 5 sliding plus 1 full layer and 8 target features. No North geometry is selected.
- A random one-layer North-shaped smoke artifact passed an isolated exact-INT4 TP=2 construction/load gate. Both ranks proved block 24 → extractor entry 25, a 2048-wide auxiliary tensor, a 2048-wide draft `fc` input, and actual one-layer context-KV construction with 4 global KV heads × 128 dimensions. The runtime stopped before generation and grants no serving or training evidence.
- One bounded exact-INT4 extraction request proved the five-feature mapping `[1, 12, 24, 35, 46]` → `[2, 13, 25, 36, 47]`, shape `[23, 5, 2048]`, BF16 dtype, tokenizer order, and byte-identical values across both TP ranks and the connector artifact. This is deterministic Phase 2 feature evidence, not a training dataset.
- `configs/north-w4a16-teacher-checkpoint-identity.json` hashes Cohere's four-shard expert-only NVFP4 W4A16 checkpoint. Rocky and Bitey independently produced manifest digest `64ee97e76305fb94c28eb93f0babf91fbb335062b14c5018cf8f86130504a27a` over 19,354,738,299 bytes.
- Bitey loaded that exact checkpoint with vLLM 0.25.1 and the narrow reviewed Cohere overlay, selecting MARLIN NVFP4 MoE. All 18,432 filtered expert-bias placeholders, totaling 22,020,096 BF16 values, were exactly zero. A bounded request produced aligned BF16 `[23, 5, 2048]` features for the five-feature mapping.
- `configs/north-fp8-teacher-checkpoint-identity.json` independently pins Cohere's seven-shard FP8 checkpoint on Rocky and Bitey: 32,029,761,935 bytes and manifest digest `35812fdf32f497a558f31bbea43e7d69f8c1cd43c66530c7499de2f293ae2bb6`. Bitey loaded it with the TRITON FP8 MoE backend and produced an aligned BF16 `[23, 5, 2048]` trace. All three target traces share token IDs but are pairwise different, with divergence increasing by depth despite high cosine similarity.
- The eight-feature mapping `[1, 7, 14, 20, 27, 33, 40, 46]` → `[2, 8, 15, 21, 28, 34, 41, 47]` passed for AutoGPTQ, W4A16, and FP8 with BF16 `[23, 8, 2048]` artifacts. AutoGPTQ TP ranks and no-prefix FP8 repeats were byte-identical. Stock W4A16/MARLIN exhibited bounded route-order noise (RMSE 0.04082/max 1.5); the pinned PR #48032 backport made both small and radix alignment deterministic and produced byte-identical exact-teacher repeats.

### Not yet evidence

- Separate live AutoGPTQ TP=2, deterministic W4A16/MARLIN, and FP8/TRITON server-to-ring-to-optimizer transactions pass with teacher/training state separation, transactional ring acknowledgement, exact tied vocabulary preservation, and no checkpoint write. Continuous scheduling, bounded transient-file release, multi-layer draft target-feature/KV integration, optimizer/checkpoint policy, quality, acceptance, latency, memory, and throughput remain unverified. These gates advance the deterministic Phase 2/3 construction boundary but are not pilots.
- The vLLM DFlash model obtains a **draft-specific** quantization configuration and passes it to its dense draft projections, including attention projections, dense MLP projections, and the auxiliary-feature `fc` projection. This makes an FP8 draft-weight export a plausible implementation path, not a validated one. ROCm compatibility and performance for an FP8 draft remain unverified.

This plan does not turn CPU construction parity, a random probe, or a static loader reading into serving evidence.

## Sequential fail-closed phases

A phase produces a versioned evidence record or stops. A failed or missing record blocks every later phase for that target.

### 0. Freeze the evaluation contract

Before loading any target, define one versioned prompt suite, decoding policy, request mix, context-length buckets, repetitions, baseline target-only measurements, and pass/fail thresholds. The contract must specify how acceptance, accepted tokens per step, latency, throughput, resident memory, and long-context behavior are measured. Use identical methodology for all target branches, but retain outputs and reports separately.

Record the shared prompt-set identity once. Responses generated from it are target-specific: `responses/NorthINT4Target/...`, `responses/NorthW4A16Target/...`, and `responses/NorthFP8Target/...`.

**Gate:** reviewers approve the contract and its thresholds before seeing candidate results. Otherwise stop; thresholds may not be tuned after an outcome is known.

### 1. Prove each verifier identity

For `NorthINT4Target`, reverify the retained `configs/north-int4-teacher-checkpoint-identity.json` immediately before extraction and serving evaluation. It must still match the exact config, index, and all declared shards.

For `NorthW4A16Target`, reverify `configs/north-w4a16-teacher-checkpoint-identity.json` and require the image/extension identity in `configs/north-w4a16-deterministic-marlin-runtime.json` before extraction, training, or evaluation. Bitey's patched CUDA MARLIN path and Rocky's ROCm fallback are distinct verifier distributions.

For `NorthFP8Target`, reverify `configs/north-fp8-teacher-checkpoint-identity.json` using the same bounded, no-tensor-loading method. Record the deployed model location, serving image/revision, quantization configuration, tokenizer identity, and tensor-parallel topology for all targets.

**Gate:** hash or identity mismatch, missing declared files, incompatible tokenizer, changed serving revision, or an unrecorded quantization change stops that branch. Do not replace the target with BF16 to continue.

### 2. Establish exact feature extraction and target handoff

Implement and test one target at a time. For each target branch:

1. Prove the mapping between zero-based transformer block IDs and exported hidden-state entries, including whether an embedding output occupies index zero. The current five-feature reference spread and the eight-feature candidate spread are not proof of Cohere2Moe indexing.
2. Prove tensor-parallel completeness: every requested layer is present exactly once in the assembled feature sequence, has the expected width, preserves token order and request boundaries, and has no missing or duplicate shard/rank contribution.
3. Prove selected-state order, clean positions `0..C-1`, and concatenated feature width match the draft `fc` input for the selected candidate.
4. Preserve the proven tied vocabulary handoff: all three targets use the byte-identical `model.embed_tokens.weight` for both input and output, with no separate LM-head tensor. Runtime target/draft objects must remain identical and frozen during draft-only optimization.
5. Preserve the audited `<MASK_TOKEN>` ID `1` behavior: masked slots use frozen shared embedding row 1. A separately trained mask embedding is not part of the current candidate and requires a new reviewed experiment.
6. Prove target-feature/KV alignment and the serving attention path before claiming that the CPU eager visibility oracle, FlexAttention predicate, or reference Qwen adapter applies to North.

**Gate:** capture a small deterministic trace containing token IDs, positions, target layer IDs, extractor IDs, feature shapes/dtypes, TP assembly evidence, embedding/LM-head identity, mask embedding evidence, and logits. Any mismatch stops that target; no training may begin.

### 3. Generate matched training examples online

For the active target only, use that exact verifier to generate the response tokens and perform teacher forwards that supply the selected hidden states. AutoGPTQ, W4A16, and FP8 examples and features feed only their corresponding branches.

Do not materialize a full hidden-state corpus. Feed detached states directly to training or retain only a bounded ring buffer sized and documented for the active job. Store response tokens and prompt/result metadata by target, but discard feature tensors after their bounded online lifetime. Disable vLLM prefix caching for teacher extraction: a controlled FP8 repeat showed that the cache-hit path can emit a different feature artifact instead of recomputing the exact requested states.

**Gate:** the run ledger must join every batch to the active verifier identity, target-specific response set, target-layer order, disabled-prefix-cache setting, and online/ring-buffer policy. Presence of mixed-target features, unknown target identity, enabled prefix caching, or an unbounded feature cache stops the run. Byte repeatability is required for all three pinned runtimes. W4A16/MARLIN additionally requires the PR #48032 runtime identity; tolerance-only stock-Marlin runs are diagnostics.

### 4. Run a 100–1,000-example pilot for each candidate

For each target, start two BF16 pilot runs from the same recorded base/random initialization and evaluate both retained candidates:

- `acceptance_first_qwen3_coder_shaped`: 8 full draft layers and 5 target features.
- `long_context_memory_qwen36_shaped`: 6 layers, 5 sliding plus 1 full layer, and 8 target features.

The long-context candidate remains blocked until the selected training and serving attention backend demonstrably enforces its sliding-window behavior. The current eager reference adapter does not establish that behavior.

Measure matched acceptance and accepted tokens per step, draft latency, verifier latency, end-to-end throughput, resident memory, and long-context behavior under the frozen contract. Preserve loss and stability traces as supporting evidence, not as a substitute for serving acceptance.

**Gate:** select a geometry separately for each target only after both candidates have comparable measurements. A blocked candidate is reported as blocked, not treated as a loss. Do not select either candidate because of estimated parameter count, a CPU forward, or results from the other verifier.

### 5. Scale the selected BF16 branch

Scale only the selected geometry for one target at a time. Final adaptation must continue using online hidden states and responses generated by that same target. Retain resumable training state, source and destination identities, data/prompt partitions, and validation traces under the target-specific artifact name.

An explicitly labeled cross-target warm-start experiment may enter here, but it is a new destination-target run with its own online final-adaptation interval. Do not resume an INT4-trained state, skip FP8 adaptation, and label the result FP8 matched.

**Gate:** promote a run to matched serving integration only when it satisfies the predeclared training-health and held-out matched-acceptance threshold, with reproducible checkpoint and evidence manifests. Divergence, identity drift, or a missing final target-specific adaptation interval fails closed to the prior checkpoint or a new pilot.

### 6. Integrate matched serving with resident weights

Integrate the BF16 artifact with only its exact verifier, first in a stopped or isolated test harness and then in the intended serving topology. Validate the target auxiliary-state path, feature width/order, context-KV preparation, target embedding/LM-head sharing or intended handoff, mask embedding behavior, and draft loading against the artifact manifest.

Rocky serving gates require resident target and draft weights. Do not use CPU or UVA offload to pass a latency, throughput, or memory gate. Record peak and steady-state resident memory separately from host storage.

**Gate:** an integration failure, fallback to a different target, weight duplication contrary to the artifact contract, or any offload-based result blocks matched acceptance. Roll back to the last verified serving configuration.

### 7. Measure matched acceptance and cross-pair diagnostics

Evaluate the three canonical BF16 drafts in the fixed 3×3 matrix below. All cells use the same frozen contract, but only diagonal evidence is eligible for promotion.

| Draft checkpoint | AutoGPTQ verifier | W4A16 verifier | FP8 verifier |
| --- | --- | --- | --- |
| `North-DFlash-NorthINT4Target-BF16Draft` | **matched: promotion evidence** | diagnostic only; never promotable | diagnostic only; never promotable |
| `North-DFlash-NorthW4A16Target-BF16Draft` | diagnostic only; never promotable | **matched: promotion evidence** | diagnostic only; never promotable |
| `North-DFlash-NorthFP8Target-BF16Draft` | diagnostic only; never promotable | diagnostic only; never promotable | **matched: promotion evidence** |

For every cell, report acceptance, accepted tokens per step, draft latency, verifier latency, end-to-end throughput, memory, and long-context behavior. The off-diagonal cells diagnose distribution sensitivity and regression risk; they cannot turn either draft into a cross-target deployment candidate.

**Gate:** a diagonal checkpoint must meet its target-specific acceptance and serving thresholds. An impressive off-diagonal result, pooled aggregate, or target-only metric cannot override a failed diagonal.

### 8. Evaluate optional FP8 draft-weight exports

Only after a matched BF16 checkpoint passes Phase 7, produce its corresponding FP8 draft-weight export and run an A/B evaluation with the same verifier and contract:

- `North-DFlash-NorthINT4Target-BF16Draft` versus `North-DFlash-NorthINT4Target-FP8Draft` on `NorthINT4Target`.
- `North-DFlash-NorthW4A16Target-BF16Draft` versus `North-DFlash-NorthW4A16Target-FP8Draft` on `NorthW4A16Target`.
- `North-DFlash-NorthFP8Target-BF16Draft` versus `North-DFlash-NorthFP8Target-FP8Draft` on `NorthFP8Target`.

Measure acceptance, accepted tokens per step, draft latency, verifier latency, end-to-end throughput, resident memory, and long-context behavior. Also record ROCm load/runtime compatibility and stability. The verifier remains unchanged throughout each comparison; only draft-weight precision changes.

**Gate:** FP8 is optional. Promote its export only if it preserves the matched acceptance requirement and wins the predeclared deployment objective without violating memory, long-context, or ROCm gates. Otherwise retain the BF16 canonical checkpoint and mark the FP8 export rejected or experimental.

### 9. Promote, monitor, and roll back

Promotion is per diagonal pair and per draft-weight precision. The promotion record must reference the exact verifier identity, artifact manifest, chosen geometry, matched evaluation report, and serving configuration. It must name a rollback artifact: the previous approved artifact for that same verifier, or target-only serving when none exists.

Monitor the production request mix using the contract's acceptance and latency signals. On verifier identity drift, acceptance regression, runtime incompatibility, memory pressure, or long-context failure, disable the drafter and roll back without substituting the other target's draft.

## Resource and storage policy

- **Bitey:** run one exact teacher and one draft/training state at a time. CPU and CUDA share unified physical memory, so the plan does not require simultaneous AutoGPTQ, W4A16, and FP8 teachers or parallel target branches.
- **Rocky:** serving measurements use resident weights only. CPU/UVA offload is outside acceptance evidence.
- **Features:** use online teacher forwards or a bounded ring buffer. Never cache the full hidden-state corpus.
- **Retention:** retain source identity manifests, target-specific response-token datasets, checkpoint/optimizer state needed for resume, manifests, and acceptance evidence. Do not retain transient feature tensors beyond the declared ring-buffer lifetime.
- **Shared material:** source code, infrastructure, prompt definitions, candidate definitions, and starting-weight identity may be shared. Their use must be recorded in each branch manifest; they do not merge branch identity or evidence.

## Explicit non-goals

This plan does not:

- train or promote one pooled, unconditioned AutoGPTQ+W4A16+FP8 draft;
- call a BF16 teacher equivalent to either deployed quantized verifier;
- select either geometry without target-specific measurements;
- treat a warm-start, static source audit, CPU reference forward, random artifact, or interrupted runtime-probe work as a passed serving gate;
- require concurrent quantized teachers, a full feature cache, CPU/UVA offload, or FP8 draft deployment;
- promote an off-diagonal cross-pair result.

For geometry assumptions and their current restrictions, see [North draft candidates](NORTH_DRAFT_CANDIDATES.md). For the existing model and feature-contract boundaries, see [Architecture decisions](ARCHITECTURE_DECISIONS.md) and [Implementation status](../IMPLEMENTATION_STATUS.md).
