# Train North with upstream Speculators

Use this path when training a deployable North DFlash draft. The repository's earlier custom CE/layout pipeline remains historical diagnostic code; it is not the training authority.

The training authority is [`vllm-project/speculators`](https://github.com/vllm-project/speculators) at commit `f7ec34182826bc89934ce710283421778022b74d`. The validated container is `local/vllm-openai-cuda:v0.25.1-north-dflash-speculators-f7ec3418-r4`, image ID `sha256:35e531ee8b9d594555af39abe4a498294592e6501c64e2d935ee99fcbd942977`.

## Contract

The validated FP8 configuration follows the upstream tutorial and implementation:

- On-policy, pretokenized prompt and response sequences.
- Auxiliary target boundaries `[2, 24, 46]`; `launch_vllm.py` appends final boundary `49` for verifier logits.
- Five Qwen3 draft layers with 2,048-token sliding attention.
- Block size 8 and `sample_from_anchor=false`.
- A 32K frequency-selected draft vocabulary.
- KL loss, DFlash decay gamma 4, and standard hidden-state noise `0.05`.
- AdamW with learning rate `3e-4` for the initial pilot.

Boundary 49 is not a fourth draft feature. Upstream training separates it as `verifier_last_hidden_states`, applies the verifier's final RMSNorm, and constructs target logits through the verifier output projection.

## Bounded execution on Bitey

Bitey's CPU and GPU allocations share one memory pool. Do not keep the teacher and trainer resident together.

1. Prepare a bounded on-policy chunk.
2. Start the exact FP8 teacher and extract transient features.
3. Stop the teacher and allow a cooldown.
4. Train with upstream `scripts/train.py`.
5. Publish and verify an explicit checkpoint epoch.
6. Serve that exact checkpoint for acceptance measurement.
7. Hash and release the transient feature files.

### Prepare pretokenized data

Each JSONL row must contain aligned `input_ids` and `loss_mask` arrays. Prompt positions use `0`; target-generated response positions use `1`.

```bash
IMAGE=local/vllm-openai-cuda:v0.25.1-north-dflash-speculators-f7ec3418-r4
MODEL=/home/douglasbrown/Models/North-Mini-Code-1.0-fp8
RUN=/home/douglasbrown/Serve/hosts/bitey/training/north-dflash/runs/<immutable-run-id>

docker run --rm --network none \
  -v "$MODEL:/model:ro" -v "$RUN:/run" \
  --entrypoint python3 "$IMAGE" \
  /opt/speculators-f7ec3418/scripts/prepare_data.py \
  --model /model --data /run/source.jsonl --output /run/data \
  --seq-length 8192 --seed 0 --num-preprocessing-workers 1 \
  --minimum-valid-tokens 1
```

Do not use `--overwrite` in an evidence directory.

### Launch extraction

```bash
docker run --rm --name north-fp8-speculators-teacher \
  --gpus all --network host --ipc host --shm-size 16g \
  --ulimit memlock=-1 --security-opt seccomp=unconfined \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -v "$MODEL:/model:ro" -v "$RUN/features:/features" \
  --entrypoint python3 "$IMAGE" \
  /opt/speculators-f7ec3418/scripts/launch_vllm.py /model \
  --hidden-states-path /features --target-layer-ids 2 24 46 -- \
  --host 127.0.0.1 --port 8095 \
  --served-model-name north-fp8-speculators-official \
  --tensor-parallel-size 1 --max-model-len 8192 \
  --max-num-seqs 1 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.50 --dtype bfloat16 --enforce-eager \
  --no-enable-prefix-caching
```

The launcher also disables chunked prefill. Generate the bounded feature chunk with the official offline generator, then stop the teacher gracefully:

```bash
docker run --rm --network host \
  -v "$RUN:/run" -v "$RUN/features:/features" \
  --entrypoint python3 "$IMAGE" \
  /opt/speculators-f7ec3418/scripts/data_generation_offline.py \
  --model north-fp8-speculators-official \
  --endpoint http://127.0.0.1:8095/v1 \
  --preprocessed-data /run/data --output /run/features \
  --concurrency 1 --validate-outputs --fail-on-error \
  --request-timeout 300 --max-retries 0

docker stop --timeout 60 north-fp8-speculators-teacher
```

### Train with the upstream model

```bash
EPOCHS=5

docker run --rm --gpus all --network none --ipc host --shm-size 16g \
  --ulimit memlock=-1 \
  -v "$MODEL:/model:ro" -v "$RUN:/run" \
  --entrypoint python3 "$IMAGE" \
  /opt/speculators-f7ec3418/scripts/train.py \
  --verifier-name-or-path /model \
  --data-path /run/data --hidden-states-path /run/features \
  --save-path /run/checkpoints --speculator-type dflash \
  --num-layers 5 --draft-arch qwen3 \
  --target-layer-ids 2 24 46 \
  --draft-vocab-size 32000 --token-freq-path /run/data/token_freq.pt \
  --mask-token-id 1 --block-size 8 --sliding-window 2048 \
  --draft-attn-impl simple_flex_attention \
  --train-data-ratio 0.9 --on-missing raise \
  --hidden-states-dtype bfloat16 --total-seq-len 8192 \
  --noise-std 0.05 --max-anchors 3072 \
  --loss-fn kl_div --dflash-decay-gamma 4.0 \
  --optimizer adamw --lr 3e-4 --weight-decay 0.01 \
  --epochs "$EPOCHS" --checkpoint-freq "$EPOCHS" \
  --no-resume-from-checkpoint --seed 0 --deterministic-cuda
```

At the pinned commit, `--max-steps` did not stop the outer epoch loop in a diagnostic run. Use `--epochs` as the hard execution bound. Also verify the checkpoint epoch: an in-memory final metric does not belong to an earlier checkpoint merely because that checkpoint exists.

Standard Speculators checkpoints include verifier vocabulary tensors. Treat them as diagnostic/deployment exports until a draft-only adapter removes those tensors under the retained-checkpoint contract.

## First end-to-end result

Evidence is under `20260726T180000Z-fp8-speculators-official-single` and summarized in [`north-fp8-speculators-official-v1.json`](../configs/north-fp8-speculators-official-v1.json).

A published, noise-free, dense-anchor epoch-99 checkpoint reached offline validation loss `0.00525763`, full top-1 accuracy `0.9980315`, and EAL `6.96662` on its duplicated seen sample. vLLM loaded the standard Speculators checkpoint directly and accepted 45 of 574 drafted tokens over 128 output tokens:

- Draft-token acceptance: `7.8397%`.
- Mean emitted tokens per speculative step: `1.54878`.
- Accepted tokens by position: `[8, 7, 6, 6, 6, 6, 6]`.

This is an end-to-end construction and parity pass. It is not held-out quality evidence and the noise-free checkpoint must not become a candidate.

## On-policy coding-data scaling

Fresh pilots then used exact FP8 target responses for Magicoder coding prompts, standard `noise_std=0.05`, and five epochs. Source rows 500–599 remained untouched for a fixed 100-prompt acceptance gate. Every gate generated 128 tokens per prompt with greedy sampling and `ignore_eos=true`.

| Training rows | Training tokens | Validation EAL | Draft acceptance | Mean emitted length | Accepted by position | Warm output tok/s |
| ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 500 | 379,567 | 0.295 | 4.73% | 1.331 | `[2725,359,61,19,0,0,0]` | 30.83 |
| 1,000 | 767,216 | 0.489 | 7.45% | 1.521 | `[3328,823,157,47,9,2,1]` | 34.50 |
| 2,000 | 1,539,673 | 0.677 | 10.44% | 1.731 | `[3640,1244,332,121,32,11,7]` | 38.59 |
| 5,000 | 3,822,908 | 0.904 | 14.90% | 2.043 | `[3714,1648,666,278,131,60,24]` | 44.57 |

The exact same holdout improved monotonically. The 5,000-row pilot accepted 6,521 tokens and produced nonzero acceptance at every draft position. All four candidates emitted byte-identical token-ID sequences across all 12,800 holdout output tokens (root `5d291d95…e1056`). Its mean emitted length exceeds the upstream tutorial's reported 1.47 for a 5K Qwen pilot, although throughput across different hardware and targets is not comparable.

Responses were bounded at 512 generated tokens; 4,451 of the 5,000 rows ended at that limit. These are valid exact on-policy prefixes, but a later quality-oriented corpus should include longer completed responses. The acceptance gates do not evaluate response quality.

Detailed identities and immutable run paths are in [`north-fp8-speculators-code-scaling-v1.json`](../configs/north-fp8-speculators-code-scaling-v1.json).

## Rocky FP8 portability

The exact 5K checkpoint also loaded with Rocky's exact FP8 target under TP=2, TRITON FP8 MoE, `TRITON_ATTN`, and `NCCL_PROTO=Simple`. Its mean emitted length was `2.035`, closely matching Bitey's `2.043`.

On the fixed 100-prompt holdout, warmed sequential target-only throughput was `28.57` output tok/s and DFlash throughput was `44.02` tok/s: an actual `1.541×` speculative speedup. A repeated fixed-prompt load gate measured:

| Concurrency | Target-only tok/s | DFlash tok/s | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 29.16 | 34.08 | 1.169× |
| 2 | 55.19 | 66.25 | 1.200× |
| 4 | 113.50 | 134.32 | 1.183× |

At 20,659 prompt tokens, both target-only and DFlash retrieved the correct NIAH answer. Acceptance fell to mean emitted length `1.460`, however, and DFlash was 2.9% slower (`16.36` versus `16.85` tok/s). Long-context adaptation or a different draft geometry is therefore required before enabling DFlash indiscriminately at long context.

Greedy target-only and DFlash outputs were not bitwise identical on Rocky: only 10/100 complete responses matched, with a median common prefix of 48.5 tokens. Target execution under speculative verification uses different shapes and can change finite-precision decisions, but the difference remains an explicit numerical parity caveat rather than an assumed harmless effect.

The legacy `NORTH_DFLASH_TIED_OUTPUT_CONTRACT` assertion must not be enabled for standardized 32K Speculators checkpoints. Those checkpoints intentionally carry a distinct pruned draft LM head; the assertion applies only to the historical full-vocabulary tied-output draft.

Detailed evidence is in [`north-fp8-speculators-rocky-portability-v1.json`](../configs/north-fp8-speculators-rocky-portability-v1.json).

## Full-prefill versus incremental features

For identical token IDs on one server, full-prefill and incremental extraction produced these differences:

| Boundary | RMSE | Maximum absolute difference |
| --- | ---: | ---: |
| 2 | 0 | 0 |
| 24 | 0.067614 | 5.625 |
| 46 | 0.273400 | 48.0 |
| 49 | 0.321915 | 15.0 |

The states drift with depth even though their token IDs match. This does not indicate an indexing failure: the official pipeline produced accepted runtime tokens. It does mean that diverse on-policy data and the standard `noise_std=0.05` augmentation are required before evaluating held-out acceptance.
