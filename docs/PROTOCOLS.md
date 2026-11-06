# Evaluation protocols

This document separates verification pairs, identification galleries, and inference routes. Record all three when reporting a result.

## Input and identity conventions

Processed MAT files contain `img` and `mask` as 2-D arrays. Image values in `[0,1]` are scaled to `[0,255]` by the loader. Foreground masks define valid grid nodes. No minutiae field is loaded.

A 32-pixel patch and 16-pixel stride define the grid. Patch centers are normalized for sampling CNN features. Nodes whose patches have less than 50% foreground coverage are masked. The CPU fixture has image shape `[1,1,640,480]` and 1,131 grid nodes; selected landmark count is 50.

UWA filenames use `<subject>_<finger>_<session>_<view>.mat`. A finger identity combines subject and finger, shared across sessions and viewpoints. CFPose filenames use `<finger>_<view>.mat`, with views numbered 1–10. Preserve these names. Mixed-domain identity handling also uses the `unsw` and `cfpose` directory names.

## UWA

The manuscript partition assigns subjects 1–100 to testing and 101–150 to training. Evaluation compares session 2 queries against session 1 reference images, including every cross-session pair in the test partition.

The manuscript describes a nominal `3,000 × 3,000` matrix: 9,000 genuine and 8,991,000 impostor pairs. The evaluator derives the actual matrix size from the supplied files. Report the numbers of reference images, query images, genuine pairs, and impostor pairs with any reproduced result; an incomplete or differently preprocessed image pool changes these denominators.

For identification, query scores are ranked over reference images. A query is correct at Rank-N if a reference entry with its finger identity occurs among the first N returned entries.

## CFPose verification

Use fingers 1–120 for testing and fingers 121–140 for training, with ten views per finger in the intended complete pool. The loader orders filenames numerically before applying the partition.

Verification evaluates each unordered pair of distinct test images once. With all 1,200 test images present:

| Quantity | Count |
|---|---:|
| All verification pairs | 719,400 |
| Genuine pairs | 5,400 |
| Impostor pairs | 714,000 |

Gallery selection does not remove images from this verification pool or change EER/TAR pair construction.

## CFPose identification

The `angle_gt` option uses a canonical gallery:

1. For each test finger, select the image with the smallest absolute ground-truth in-plane refine angle.
2. Use that image as the finger's single gallery entry.
3. Exclude the gallery image from that finger's probe set; the remaining nine views are probes.

A complete test pool therefore yields 120 gallery images and 1,080 probes. Ground-truth angles are used only to select gallery entries. They are neither encoder inputs nor verification scores, and this evaluation option does not rotate the input images.

Select the protocol explicitly:

```bash
python -m MyModel.scripts.test --ckpt_path checkpoints/your_model.pt --data_name cfpose --data_root data --gallery_strategy angle_gt --gallery_metadata metadata/angle_refine_cfpose_gt.mat --device cpu --num_workers 0
```

The MAT file must contain `angle_refine`. The ground-truth array convention is `[finger, view, component]` with shape `[140,10,3]`; the evaluator reads component index 1 as the in-plane angle. Finger and view numbers in filenames map to array indices minus one. For equal absolute angles, the first image in deterministic file order is retained.

The CLI's default `gallery_strategy=first` uses the first-view gallery. The `auto`, `fixed3`, and `angle_jia` options provide additional gallery choices. Record the selected option with each result; select `angle_gt` for the canonical ground-truth-angle protocol.

## Scoring routes and metrics

The CLI reports both routes from one feature extraction by default:

| Route | Scoring in the dual-route evaluation |
|---|---|
| NTN | Learned interaction scorer applied to normalized templates |
| L2 | Cosine similarity between normalized templates, with the same ranking as increasing negative Euclidean distance |

The primary return route inherits `use_ntn` from the checkpoint unless overridden with `--use_ntn` or `--no-use_ntn`. Both-route reporting is enabled by default. The single-route helper also has a monotonic `exp(-L2)` score representation, so raw thresholds must be associated with the exact scoring route.

An NTN-trained checkpoint evaluated with L2 remains an NTN-trained model. The manuscript's no-NTN ablation changes training supervision and must be identified separately.

EER uses the empirical ROC point minimizing `|FPR − FNR|`, then averages FPR and FNR at that point. TAR at FAR = 0.01% is the TPR at the last empirical ROC point with `FPR ≤ 1e-4`. The implementation uses this empirical rule without ROC interpolation.

## Checkpoint selection

The supplied configurations use UWA subjects 101–150 and all 200 CFPose training images for fitting. Periodic evaluation uses the test partitions. Stage I evaluates every ten epochs and selects by test EER; Stage II evaluates CFPose every five epochs and selects by CFPose EER. Keep the checkpoint-selection history with the training and evaluation output.

