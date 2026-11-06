# Third-party notices

## HyperGCT

`MyModel/model/HyperGCTcore.py` adapts hypergraph components from **HyperGCT: A Dynamic Hyper-GNN-Learned Geometric Constraint for 3D Registration**, by Xiyu Zhang, Jiayi Ma, Jianwei Guo, Wei Hu, Zhaoshuai Qi, Fei Hui, Jiaqi Yang, and Yanning Zhang, ICCV 2025.

Source publication: [official CVF paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_HyperGCT_A_Dynamic_Hyper-GNN-Learned_Geometric_Constraint_for_3D_Registration_ICCV_2025_paper.pdf).

The archived upstream source carries the MIT License:

> Copyright (c) 2024 zhangxy0517

The complete copyright and permission notice is retained in [licenses/HyperGCT-LICENSE.txt](licenses/HyperGCT-LICENSE.txt). It applies to the upstream code and its substantial derived portions. The project-level MIT license does not replace this notice.

The adapted file includes hypergraph feature aggregation, non-local interaction, and dynamic graph-update components. Project-specific fingerprint encoding, foreground-grid input, landmark readout, training integration, and release changes are distributed alongside those components.

```bibtex
@InProceedings{Zhang_2025_ICCV,
  author = {Zhang, Xiyu and Ma, Jiayi and Guo, Jianwei and Hu, Wei
            and Qi, Zhaoshuai and Hui, Fei and Yang, Jiaqi and Zhang, Yanning},
  title = {HyperGCT: A Dynamic Hyper-GNN-Learned Geometric Constraint
           for 3D Registration},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  month = {October},
  year = {2025},
  pages = {24750--24759}
}
```

## Runtime dependencies and pretrained weights

PyTorch, torchvision, NumPy, SciPy, Pillow, Matplotlib, pandas, scikit-learn, tqdm, and TensorBoard retain their respective licenses. They are installed as dependencies rather than vendored into this repository.

The training encoder uses torchvision's ImageNet ResNet-18 weights by default. Those weights are obtained through torchvision and are not included in this package. The project's MIT license covers project code; it does not grant rights to dataset images, biometric records, external annotations, or separately obtained pretrained weights.

## Other archived projects

GraSP and Deep Graph Matching Consensus were present as separate projects in the development archive. Their source code is not included in this release, and the released fingerprint model does not import them.

