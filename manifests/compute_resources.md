# Compute Resources

The audit was run on data-center NVIDIA GPUs with 32 GB to 80 GB class memory
for standard jobs and larger-memory GPUs for the most expensive PGD sweeps.
Exact runtime depends on encoder throughput, image preprocessing, batch size,
and the number of clean-correct samples.

## Reported Resource Classes

- Linear-head training: GPU jobs with frozen features or frozen encoders.
- Canonical `l2` audit: PGD-10 x 5, PGD-50 x 10, and PGD-200 x 5 records on
  the 42-encoder pool.
- `linf` audit: PGD-10 x 5 and PGD-100 x 5 records over the 42-encoder pool
  and radius sweep summaries.
- Deep-reference diagnostics: PGD-500 x 5 fields where recorded.
- Intervention runs: LoRA + LLR and coordinate-targeted cells for the
  reported ViT-B/16 and Swin-B comparisons.

## Reproducibility Notes

The package reports protocol configuration, derived records, and scripts for
recomputing statistics. It does not include cluster logs because those logs can
contain local scheduler, node, and path metadata. Users reproducing the full
audit should size runtime roughly linearly in the number of encoder, radius,
step, and restart combinations.
