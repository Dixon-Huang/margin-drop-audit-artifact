# Intervention Record Schema

The intervention table is
`records/intervention/intervention_records.csv`.

- `encoder`: Encoder or backbone display name.
- `method`: Baseline or intervention method.
- `record_type`: Either `closure` for endpoint robust-retention records or
  `coordinate_delta` for coordinate movement records.
- `budget`: Evaluation budget or budget family.
- `R10`, `R50`, `R200`: Pairwise robust retention at the stated budgets when
  available.
- `collapse_10_200`: Retention gap from PGD-10 to PGD-200.
- `delta_lambda`: Median movement in margin slack.
- `delta_kappa`: Median movement in on-ray shortfall.
- `delta_rho50`: Median movement in the PGD-50 drift coordinate.
- `delta_m`: Median per-sample movement in survival margin.
- `source`: Internal provenance note for the record values.

Rows with endpoint metrics leave coordinate-delta fields blank. Rows with
coordinate deltas leave endpoint fields blank when the endpoint is recorded in
a separate row.
