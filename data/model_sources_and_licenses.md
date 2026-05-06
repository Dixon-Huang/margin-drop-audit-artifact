# Model Sources And License Notes

The supplementary package does not redistribute pretrained checkpoints or
linear-head checkpoints. Encoder identifiers and source packages are listed in
`encoder_pool.csv`.

## Source Families

- `timm / torchvision`: Models are loaded from public `timm` or torchvision
  registries. Users should follow the upstream model-card and checkpoint
  licenses for each identifier.
- `open_clip`: CLIP-family checkpoints are loaded from public OpenCLIP or
  compatible registries. Users should follow the upstream checkpoint license
  and dataset-use terms.
- Self-supervised DINO and DINOv2 models: Users should obtain checkpoints from
  the public provider or from the source package listed in `encoder_pool.csv`.

## Dataset Note

Raw ImageNet images are not redistributed. `imagenet100_class_split.csv`
records the sample and class identifiers observed in the audit records. Users
must obtain ImageNet under its own access terms.

## Checkpoint Note

The column `head_checkpoint_pattern` records the naming convention used by the
audit. It is intentionally relative and does not point to a local user path.
