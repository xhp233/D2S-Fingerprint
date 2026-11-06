# Dense-to-Sparse Topological Distillation

Code accompanying **Dense-to-Sparse Topological Distillation for Minutiae-Label-Free Contactless Fingerprint Matching**.

**Houpu Xu · Yuwei Jia · Zhe Cui · Yuan Dong**

The model builds a dense foreground hypergraph from a fingerprint image and mask, applies dynamic hypergraph contraction, and selects 50 landmarks to produce a 128-dimensional unit template. Relational Structural Interaction supplies pairwise supervision during training. The resulting checkpoint supports both NTN scoring and efficient template matching with the L2 route.

## Installation

Use Python 3.11+ with the dependencies in `requirements.txt`. Run commands from the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/smoke_cpu.py
python -m unittest discover -s tests -p "test_*.py"
```

The CPU smoke runs a synthetic MAT through `640 × 480 image → 1,131 grid nodes → 50 selected nodes → 128-D unit template`. The tests cover the evaluation interface, architecture, configuration handling, checkpoint loading, and separation of gallery selection from verification. See the [training and evaluation guide](docs/REPRODUCIBILITY.md) for environment and usage details.

## Data and evaluation

Obtain the datasets and preprocessing assets from their owners, then arrange the processed MAT files as follows:

```text
data/
├── unsw/raw_data_640/<subject>_<finger>_<session>_<view>.mat
└── cfpose/raw_data_640_CFPose/<finger>_<view>.mat
```

Each MAT file contains two 2-D arrays, `img` and `mask`. The loader uses images and foreground masks without minutiae labels. Dataset images, ground-truth angle annotations, and model checkpoints are supplied separately by their owners and are not bundled here.

Evaluate a compatible local checkpoint on UWA:

```bash
python -m MyModel.scripts.test --ckpt_path checkpoints/your_model.pt --data_root data --data_name unsw --device cpu --num_workers 0 --report_both_routes
```

For CFPose, select a canonical gallery with a local ground-truth refine-angle file:

```bash
python -m MyModel.scripts.test --ckpt_path checkpoints/your_model.pt --data_root data --data_name cfpose --gallery_strategy angle_gt --gallery_metadata metadata/angle_refine_cfpose_gt.mat --device cpu --num_workers 0 --report_both_routes
```

CFPose verification uses all unordered image pairs. The command above uses one canonical gallery image per finger, selected by minimum absolute ground-truth in-plane refine angle, and the other nine images as probes. The angles define this gallery only. The command-line default `first` uses the first-view gallery; specify `angle_gt` for canonical-gallery identification. Details are in [protocols](docs/PROTOCOLS.md).

The evaluator reports NTN and L2 routes from the same checkpoint by default. Using the L2 route on an NTN-trained checkpoint changes inference scoring; the no-NTN training ablation is a separate experiment.

## Results

All values are percentages. TAR is measured at FAR = 0.01%.

| Dataset | Route | EER ↓ | TAR ↑ | Rank-1 ↑ | Rank-5 ↑ | Rank-10 ↑ |
|---|---|---:|---:|---:|---:|---:|
| UWA | NTN | 5.62 | 26.18 | 82.33 | 91.08 | 93.73 |
| UWA | L2 | 7.46 | 26.15 | 83.27 | 92.35 | 94.60 |
| CFPose | NTN | 3.65 | 35.33 | 86.39 | 98.15 | 99.26 |
| CFPose | L2 | 3.93 | 35.35 | 86.57 | 98.15 | 99.35 |

## Training configurations

[configs/archived_stage1_uwa.json](configs/archived_stage1_uwa.json) configures UWA pre-training. [configs/archived_stage2_mixed.json](configs/archived_stage2_mixed.json) configures mixed UWA+CFPose fine-tuning with feature constancy. Dataset and checkpoint paths can be supplied through command-line arguments. See the [training and evaluation guide](docs/REPRODUCIBILITY.md) for commands and configuration details.

## License and attribution

Project code is released under the [MIT License](LICENSE). The hypergraph backbone adapts HyperGCT; its MIT copyright and permission notice are retained in [licenses/HyperGCT-LICENSE.txt](licenses/HyperGCT-LICENSE.txt). See [third-party notices](THIRD_PARTY_NOTICES.md).


## Citation

Author and title metadata are provided in [CITATION.cff](CITATION.cff).
