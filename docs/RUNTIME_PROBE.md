# North DFlash runtime probe — random, non-production only

`north-dflash runtime-probe-generate` creates the last artifact permitted before a
GPU runtime probe: a random BF16 Hugging Face `DFlashDraftModel` checkpoint.
It does **not** read North weights, tokenize data, train, start a server, or
measure acceptance. Its only purpose is to determine whether the North target
auxiliary hidden-state path and vLLM DFlash loader can construct and load.

## Guarded artifact generation

Inspect the default bounded geometry without writing weights:

```bash
PYTHONPATH=src .venv/bin/python -m north_dflash_training.cli runtime-probe-config
```

The default `smoke` geometry has one draft layer, one selected target feature
(block 24), and block size two (one clean anchor plus one masked future). It
still preserves the runtime boundary: hidden size 2048; dense DFlash MLP 6144;
32 Q heads; 4 KV heads; head dim 128; vocab/draft vocab 262144; mask ID 1; 49
target layers; North RoPE max position/theta (500000/50000); SiLU; BF16; and
RMSNorm epsilon `1e-6`.

To write it to an as-yet nonexistent directory:

```bash
PYTHONPATH=src .venv/bin/python -m north_dflash_training.cli runtime-probe-generate \
  --output /path/that/does/not/exist/north-dflash-smoke-random
```

The command seeds PyTorch deterministically (`20260724` unless overridden),
performs a conservative free-space preflight, refuses any existing output, and
writes `config.json`, `model.safetensors`, and `runtime-probe-manifest.json`.
The manifest is prominently random/non-production, links (without reading
weights) to the retained exact North identity manifest, records the tensor hash
and seed, and proves that no target embedding or LM-head tensor was saved.

`full` is the 8-full-draft-layer/5-feature candidate with block size 16 and is
not a default. It requires both selecting it and an intentional acknowledgement:

```bash
... runtime-probe-generate --geometry full --confirm-full-random-nonproduction --output /new/path
```

Do not run that command yet. It remains random and unsuitable for every use
other than loader/plumbing diagnosis.

## Layer IDs and loader evidence

The generated `dflash_config.target_layer_ids` are ascending **zero-based target
transformer block indices**. The default is `[24]`; full is `[1, 12, 24, 35,
46]`. vLLM's target hidden-state list has an embedding output at entry zero, so
its extractor field is explicitly stored as `eagle_aux_hidden_state_layer_ids`
with `target_layer_id + 1`: `[25]` for smoke and `[2, 13, 25, 36, 47]` for full.
This is not an inferred alternate selection.

Before generating, the tool fails closed unless the local vLLM checkout still
proves all of the following:

- registry maps `DFlashDraftModel` to `qwen3_dflash.DFlashQwen3ForCausalLM`;
- the loader prefixes reference checkpoint tensors with `model.`, applies the
  Q/K/V and gate/up stacking mapper, skips absent `embed_tokens`, and builds
  fused context-KV buffers;
- the DFlash runtime shares target embedding/LM-head when the draft has no
  own-copy flag; and
- the runtime fallback converts each DFlash target ID to extractor ID `+1`.

The reference `dflash.model.DFlashDraftModel.save_pretrained(...,
safe_serialization=True)` is used rather than a hand-written state dict. Its
state contains only `fc`, `hidden_norm`, `norm`, and `layers` tensors—no
embedding or LM head. Tiny CPU fixture tests save and reopen the safetensors
file and verify those names before a North-shaped artifact is allowed.

## Bitey runtime-probe criteria (do not launch from this repository)

A future Bitey operator may use this artifact only in an isolated, stopped
service/test harness after all of these criteria are recorded:

1. The deployed target is the exact identity linked in the artifact manifest;
   do not substitute a BF16/dequantized target or a different tokenizer.
2. vLLM source/revision matches the manifest's loader evidence. Re-run the
   static check if the image or checkout changed.
3. The DFlash config parses as `DFlashDraftModel`; vLLM selects the DFlash
   loader; and loading consumes only the artifact's draft tensors while target
   embedding/LM-head are shared (not duplicated).
4. The target exports auxiliary states at `[25]` for smoke (or the recorded
   full list), each width 2048. The concatenated width must match `fc` input:
   2048 for smoke and 10240 for full.
5. Context-KV precomputation constructs exactly one draft KV layer for smoke
   (eight for full) with 4 KV heads × 128 dimensions, and no shape/index error.
6. Stop **before** request serving, token acceptance/rejection, output quality,
   throughput, optimization, or checkpoint mutation. A successful probe is
   only construction/load plus the aux-state/KV shape evidence.

Failure at any point is a runtime-integration finding, not permission to add
weights, train, alter the target, or run an acceptance experiment.
