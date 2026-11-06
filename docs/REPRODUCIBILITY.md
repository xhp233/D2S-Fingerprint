# Training and evaluation guide

The model combines a layer3 ResNet-18 feature encoder with dynamic hypergraph contraction, landmark selection, and relational structural supervision. It produces 128-dimensional unit templates from fingerprint images and foreground masks.

## Environment

The paper uses PyTorch 2.1.0 on Ubuntu 22.04 with an NVIDIA RTX 3090 (24 GB). Dependency versions for PyTorch 2.1.0 are listed in `requirements-legacy-reference.txt`. The `requirements.txt` file provides a newer PyTorch 2.9.1 environment. The CPU-check environment is recorded in [tested_environment.json](tested_environment.json).

Run installation and checks from the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/smoke_cpu.py
python -m unittest discover -s tests -p "test_*.py"
python -m MyModel.scripts.train --help
python -m MyModel.scripts.test --help
```

The synthetic MAT smoke exercises the dataset loader, CNN, hypergraph, landmark selection, and template readout:

| Item | Output |
|---|---|
| Input tensor | 1 x 1 x 640 x 480 |
| Grid nodes | 1,131 |
| Selected landmarks | 50 |
| Template | 1 x 128, finite, L2 norm 1.0 |

The unit tests cover the ten-field benchmark interface, empty-input behavior, layer3 architecture, configuration precedence, unknown-key handling, gallery selection, and checkpoint metadata loading.

## Data preparation

Arrange processed MAT files according to [protocols](PROTOCOLS.md). Each file supplies an image and foreground mask as `img` and `mask`. Preserve the dataset filename conventions so the loader can recover finger identity, session, and viewpoint.

## Two-stage training

```bash
python -m MyModel.scripts.train --config configs/archived_stage1_uwa.json --data_root data
python -m MyModel.scripts.train --config configs/archived_stage2_mixed.json --data_root data
```

Stage I trains on UWA and saves to `checkpoints/archived_stage1`. Stage II loads `checkpoints/archived_stage1/best.pt` for initialization and feature-constancy supervision, then saves to `checkpoints/archived_stage2`. The Stage II configuration starts a fresh optimizer with `load_weights_only=true`.

Both configurations use seed 42, batch size 32, three hypergraph layers, KNN size 25, landmark pool size 50, and a 128-D template. Stage II uses a 0.5 feature-constancy weight, a 0.5 UWA sampling ratio, and domain loss weights of 1.0 for UWA and 4.0 for CFPose. The JSON files contain the full training arguments, including maximum epochs and checkpoint intervals.

Explicit command-line values override JSON values. Unknown JSON keys are rejected. Relative paths resolve from the current working directory. Run from the repository root or supply explicit paths. `--data_root` selects the directory containing both dataset subdirectories; `HGNN_DATA_ROOT` provides the default when configuration and command line omit it. TensorBoard output defaults to `runs/tensorboard` and can be changed with `--log_root`.

Training initializes ResNet-18 from torchvision's ImageNet weights. Torchvision downloads these weights when they are absent from its cache. The CPU smoke uses `pretrained_backbone=False`; checkpoint evaluation loads the supplied fingerprint weights directly.

## Evaluation

Evaluate UWA:

```bash
python -m MyModel.scripts.test --ckpt_path checkpoints/your_model.pt --data_root data --data_name unsw --device cpu --num_workers 0 --report_both_routes
```

Evaluate CFPose with a canonical gallery:

```bash
python -m MyModel.scripts.test --ckpt_path checkpoints/your_model.pt --data_root data --data_name cfpose --gallery_strategy angle_gt --gallery_metadata metadata/angle_refine_cfpose_gt.mat --device cpu --num_workers 0 --report_both_routes
```

Use `--device cuda` for GPU evaluation. The evaluator reports EER, TAR at FAR = 0.01%, and Rank-1/5/10. See [protocols](PROTOCOLS.md) for pair construction, gallery options, and NTN/L2 scoring.

## Checkpoints and experiment records

Checkpoint loading restores the saved model arguments and parameters. The loader uses `weights_only=False` to support Python/NumPy metadata in checkpoints across PyTorch versions.

Record the effective configuration, data split, seed, checkpoint, scoring route, and gallery option alongside each result. Keep the dataset image counts and checkpoint-selection history with the evaluation output.
