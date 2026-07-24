# Architectural decisions requiring primary review

These are intentionally decisions, not hidden defaults:

- **Target identity:** exact local North `Cohere2MoeForCausalLM`, 49 layers, 2048 hidden size, 4-bit int AutoGPTQ with group size 32 and 128 experts / top-8 routing.
- **Draft shape:** five layers and block size 16 are paper-shaped candidates only. A dense draft is proposed as an investigation baseline; copying North's MoE is not justified yet.
- **Layer selection:** `[1, 12, 24, 35, 46]` is mechanically derived using the reference dflash spread formula for 49 target layers and five draft layers. Hidden-state indexing for Cohere2Moe may change this.
- **Mask token:** unresolved. The scaffold refuses to assert that an arbitrary vocabulary ID is safe. The CLI's synthetic default is not a North decision.
- **Features:** the estimator shows that a dataset-wide raw cache is costly (for 800K × 3072, 5 × 2048 BF16 features: about 45.8 TiB), while a 512-token ring is about 10 MiB for one stream. The actual online/ring design must preserve the feature alignment needed by sampled anchors.
- **Precision:** BF16 can initialize a draft if approved, but it cannot replace the exact int4 AutoGPTQ expert-only teacher or become the claimed deployment target.
- **Integration:** sparse attention, AutoGPTQ hidden-state extraction, and vLLM Cohere2Moe auxiliary state are not implemented here and must be reviewed independently.
