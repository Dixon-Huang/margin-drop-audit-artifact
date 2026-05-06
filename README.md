# Supplementary Material

This package contains anonymized code, configuration files, derived audit
records, schemas, and manifests for reproducing the reported statistics and
figures for "Margin-Drop Coordinates for Cross-Budget Robustness Audits".

The records are derived measurements on publicly available pretrained vision
encoders and ImageNet-derived evaluation inputs. Raw ImageNet images, model
checkpoints, linear-head checkpoints, logs, and manuscript sources are not
included.

## Contents

- `data/`
  - `imagenet100_class_ids.txt`: The 100 ImageNet-100 class identifiers used
    in the audit records.
  - `imagenet100_class_split.csv`: ImageNet-100 sample and class identifiers
    observed in the audit records.
  - `encoder_pool.csv`: Canonical encoder identifiers, family labels,
    checkpoint source package, and checkpoint patterns.
  - `family_labels.csv`: Family labels used by family-cluster statistics.
- `records/`
  - `canonical_l2/`: 42-encoder `l2`, `eps=0.25` coordinate records with
    PGD-10, PGD-50, PGD-200, and PGD-500 fields where available.
  - `heldout_l2/`: Held-out encoder-pool `l2` records.
  - `canonical_linf/`: 42-encoder `linf` records, sweep summaries, and
    cross-norm consistency tables.
  - `pgd500/`: Canonical coordinate records with PGD-500 fields retained for
    deep-reference checks.
  - `ce_apgd/`: Standard top-1 untargeted CE/APGD records and per-encoder
    summary table.
  - `perturbation_vectors/`: Off-ray geometry summaries used by Appendix F.
  - `intervention/`: Paper-reported intervention closure and coordinate-delta
    tables.
- `configs/`: Attack, audit, figure, and training configuration files.
- `schemas/`: Field definitions for per-sample and per-encoder records.
- `scripts/`: Entry points for audits, coordinate computation, statistics,
  figure reproduction, head training, and intervention evaluation templates.
- `src/`: Reusable implementation for margins, attacks, coordinates,
  statistics, and plotting.
- `manifests/`: Package inventory, checksums, and compute-resource notes.

## Environment

Use Python 3.10 or newer. The package was smoke-tested with Python 3.13 and
the dependency set in `requirements.txt`. A clean virtual environment is
recommended for review because unrelated packages in a shared environment can
create dependency-conflict warnings outside this package.

## Main Reproduction Commands

Set the artifact root once before running commands:

```bash
# Use this when the archive has been unpacked inside a parent directory.
export ARTIFACT_ROOT=supplementary

# Use this instead when this README is already at the repository root.
# export ARTIFACT_ROOT=.
```

Install dependencies:

```bash
pip install -r "$ARTIFACT_ROOT/requirements.txt"
```

Verify the headline statistics from the packaged tables:

```bash
python "$ARTIFACT_ROOT/scripts/compute_statistics.py" \
  --records-root "$ARTIFACT_ROOT/records" \
  --out "$ARTIFACT_ROOT/manifests/recomputed_statistics.json"
```

Generate coordinate rows from a stored audit JSON:

```bash
python "$ARTIFACT_ROOT/scripts/compute_coordinates.py" \
  --input-json audit_records.json \
  --out coordinate_records.csv
```

The converter accepts either the canonical margin keys
`M`, `Lp`, `D_fgsm`, and `D_pgd`, or the packaged raw-record aliases
`M_k_star`, `L_k_star`, `D_lower_k_star`, and `fidelity_k_star`.

Run a fixed-competitor audit with user-provided model and dataloader builders:

```bash
python "$ARTIFACT_ROOT/scripts/run_audit.py" \
  --model-builder user_builders:build_model \
  --data-builder user_builders:build_loader \
  --eps 0.25 \
  --steps 50 \
  --restarts 10 \
  --out audit_records.json
```

Train a linear head from stored feature tensors:

```bash
python "$ARTIFACT_ROOT/scripts/train_linear_head.py" \
  --train-features train_features.npz \
  --val-features val_features.npz \
  --out linear_head.pt
```

Regenerate example figure panels from the packaged records:

```bash
python "$ARTIFACT_ROOT/scripts/generate_figures.py" \
  --records-root "$ARTIFACT_ROOT/records" \
  --out-dir reproduced_figures
```

Summarize the packaged intervention records:

```bash
python "$ARTIFACT_ROOT/scripts/run_intervention.py" \
  --records "$ARTIFACT_ROOT/records/intervention/intervention_records.csv" \
  --out intervention_summary.json
```

The intervention summary writer emits JSON for `.json` outputs and CSV for
other output suffixes.

## Record Schemas

Per-sample coordinate records are documented in
`schemas/per_sample_record.md`. Per-encoder summary records are documented in
`schemas/per_encoder_summary.md`. Intervention records are documented in
`schemas/intervention_record.md`.

## Asset And License Notes

The code in this supplementary package is released under the license in
`LICENSE`. Derived audit records may be used for review and reproduction under
the terms in `LICENSE`. Raw ImageNet data is not redistributed. Pretrained
model checkpoints and linear-head checkpoints are not redistributed. The
encoder table lists the source package or checkpoint family so that users can
obtain models from the original providers under their respective licenses and
terms.

## Anonymity

The package intentionally omits author names, local absolute paths, cluster
logs, version-control metadata, notebook outputs, and credentials.
