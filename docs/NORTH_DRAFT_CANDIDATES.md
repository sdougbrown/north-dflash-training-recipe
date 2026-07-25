# North draft geometry candidates — no selection

These are two review candidates derived from the published geometry supplied
for this investigation. They are **not** a North configuration decision. Their
five- and eight-feature teacher mappings have been extracted, but neither full
draft geometry has been instantiated or trained against North. Both preserve
the read-only North shape used by this scaffold: hidden size 2048, 32 query heads, 4 KV heads,
head dimension 128, dense draft MLP width 6144, SiLU, and block size 16.
North's target expert width 768 is not the dense draft width.

The estimates below count the locally inspected `dflash.model.DFlashDraftModel`
only: Q/K/V/O, a three-matrix dense SwiGLU MLP, its two per-layer RMSNorms,
the four per-layer Q/K/input/post RMSNorms, final/hidden RMSNorms, and `fc [target_features * H, H]`. They exclude
North's shared frozen embedding and LM head, optimizer state, activations,
and any teacher feature store. They assume bias-free projections as in the
local DFlash model and BF16/FP16 weights/KV values.

| Candidate | Published-geometry input | Draft / target features | Estimated draft weights | Draft KV per retained query token | Feature staging per clean token |
| --- | --- | ---: | ---: | ---: | ---: |
| `acceptance_first_qwen3_coder_shaped` | Qwen3-Coder-30B-A3B-DFlash: 8 full draft layers, 5 target features | 8 / 5 | 473,995,264 params; 904.1 MiB BF16 | 16,384 bytes (16 KiB) | 20,480 bytes (20 KiB) |
| `long_context_memory_qwen36_shaped` | Qwen3.6: 6 draft layers, 5 SWA + 1 full, 8 target features | 6 / 8 | 373,323,264 params; 712.1 MiB BF16 | 12,288 bytes (12 KiB) | 32,768 bytes (32 KiB) |

## Trade-off review

- **Acceptance-first:** eight full layers make the draft about 27% larger in
  weights and 33% larger in retained draft KV than the six-layer candidate.
  It keeps target-feature staging at five North states and is the direct shape
  match for the supplied Coder DFlash geometry.
- **Long-context-memory:** six layers offset the larger eight-feature `fc`, so
  total draft weights and retained draft KV are lower, but each clean context
  token needs 60% more extracted feature width. Its intended attention-memory
  benefit depends on actual SWA, not merely the layer count.

## Hard acceptance gates

1. No candidate is selected in `configs/north-dflash-candidate.json`; it lists
   both as `reviewed_draft_candidates` with
   `draft_candidate_selection: none`.
2. The local DFlash reference has been exercised only with
   `config._attn_implementation == "eager"`. Its
   `Qwen3DFlashAttention` passes `sliding_window` into Transformers 4.57.1's
   `eager_attention_forward`, but that function does not apply a window.
   Therefore the 5-SWA/1-full candidate is **blocked** under this eager
   adapter. Do not infer SWA behavior or replace the committed visibility mask
   with a guessed backend integration.
3. The upstream Qwen3.6 draft order is five sliding-attention layers followed
   by one full-attention layer. This is recorded as reference geometry, not yet
   selected for North.
4. Exact AutoGPTQ, W4A16, and FP8 teachers now expose detached five- and
   eight-feature traces with verified numbering. The tied North embedding,
   mask row, output projection, and bounded in-memory consumer contracts pass.
   One-step real-draft optimization now passes for every verifier family and a
   draft-only save/resume boundary exists. The first retained FP8 micro-pilot is
   configured for the eight-full-layer candidate, but has not run and does not
   select it. Candidate comparison remains blocked on retained measurements and
   the six-layer candidate remains blocked on real sliding-window attention.
   Teacher extraction must disable prefix caching; W4A16 must use the pinned
   deterministic PR #48032 MARLIN runtime rather than the stock tolerance-only
   path.

The source-controlled generated review artifact contains the same machine
readable estimates. It is config-only; no North weights, GPU, package, or data
download was involved.
