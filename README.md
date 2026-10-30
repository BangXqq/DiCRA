# DiCRA — private code snapshot

This private repository stores the code associated with the DiCRA low-light image enhancement study. It is a **code-only archive**, not a self-contained reproduction package or a public release.

## Contents

- `archs/`, `models/`, `data/`, `losses/`, `decom_tools/`, `options/`, `scripts/`, and the root training/testing entry points contain the Diff-Retinex++ backbone code and configurations.
- `tools/` contains the local residual-correction training, evaluation, ablation, and comparison scripts used during DiCRA development. These include historical experimental variants; do not assume every script implements the final paper configuration.
- `README.upstream.md` preserves the upstream Diff-Retinex++ documentation and attribution.

## Deliberately excluded

Datasets, pretrained backbone weights, trained DiCRA/FREC weights, cached predictions, experimental results, manuscripts, figures, and local records are not included. Some scripts expect external checkpoints or local data paths and will need configuration before running. Results cannot be reproduced from this repository alone.

## Before any public release

Verify the exact paper-to-code mapping, replace local path assumptions with documented configuration, add a tested reproduction workflow and dependency versions, review third-party licensing and attribution, and decide how authorized weights and datasets will be obtained. Do not make this repository public without that review.

The Diff-Retinex++ backbone originates from [XunpengYi/Diff-Retinex-Plus](https://github.com/XunpengYi/Diff-Retinex-Plus); see `README.upstream.md` for its paper and usage information. DiCRA-specific changes and experiments are separate from that upstream project.
