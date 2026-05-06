# Per-Encoder Summary Schema

Per-encoder summaries aggregate clean-correct per-sample records by encoder.

## Canonical Fields

- `encoder`: Encoder identifier.
- `pretty_encoder`: Display name when available.
- `family`: Architecture or pretraining family.
- `n_samples`: Number of retained clean-correct samples.
- `in_raw_pool`: Whether the encoder belongs to the canonical raw pool.
- `in_analytical_pool`: Whether the encoder belongs to the non-floor
  analytical subset.

## Retention And Collapse

- `R10`, `R50`, `R100`, `R200`, `R500`: Pairwise robust retention at the
  corresponding PGD budget.
- `delta_R_10_200`: Cross-budget collapse from PGD-10 to PGD-200.
- `conditional_collapse_10_200`: Fraction of PGD-10 survivors that fail
  PGD-200.
- `conditional_collapse_10_200_pp`: Same conditional collapse in percentage
  points.

## Coordinate Summaries

- `bar_lambda`: Median margin slack.
- `bar_kappa`: Median on-ray shortfall.
- `bar_tau10`: Median net residual at PGD-10.
- `bar_rho10`: Median drift coordinate at PGD-10.
- `rho10`, `rho50`, `rho100`, `rho200`, `rho500`: Budget-specific median
  drift aliases used by some scripts.
- `mean_rho10`, `mean_rho50`, `mean_rho100`, `mean_rho200`, `mean_rho500`:
  Budget-specific means.
- `rho10_q75`, `rho10_q90`, `rho10_iqr`, `rho10_mad`: Tail and dispersion
  summaries for `rho_10`.
- `median_Lp`: Median first-order attack scale.
- `median_raw_gap_10`: Median raw PGD-over-one-step gap at PGD-10.
- `median_D_pgd10`, `median_D_fgsm`: Median measured drops.

## Protocol-Specific Tables

The `records/canonical_linf/` and `records/ce_apgd/` summaries use protocol
specific column names, such as `DeltaR_10_100`,
`delta_R_top1_10_100`, `R_top1_10`, and `R_top1_100`. These names are kept
explicit to avoid mixing pairwise fixed-competitor retention with standard
top-1 CE/APGD retention.
