<div align="center">
<!-- <h1>TopoFG</h1> -->
<h3>[AAAI 2026] Fine-Grained Representation for Lane Topology Reasoning</h3>
<h4>Guoqing Xu, Yiheng Li and Yang Yang<h4>
<h5>MAIS&CASIA, UCAS<h5>
<h6>Guoqing Xu and Yiheng Li contribute equally.<h6>
</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2511.12590)

</div>

## Introduction

This repository is an official implementation of TopoFG.

## News
- [2025/11/23] Codes and weights are released.
- [2025/11/18] Camera Ready version is released.
- [2025/11/8] TopoFG is accepted by AAAI 2026 🎉🎉.

## Environment Setting

- Linux
- Python 3.8.x
- NVIDIA GPU + CUDA 11.1
- PyTorch 1.9.1


```bash
conda create -n topologic python=3.8 -y
conda activate topologic

# (optional) If you have CUDA installed on your computer, skip this step.
conda install cudatoolkit=11.1.1 -c conda-forge

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Install other required packages.
```bash
pip install -r requirements.txt
```


## Data Preparation
Following [OpenLane-V2 repo](https://github.com/OpenDriveLab/OpenLane-V2/blob/v1.0.0/data) to download the data and run the [preprocessing](https://github.com/OpenDriveLab/OpenLane-V2/tree/v1.0.0/data#preprocess) code.



## Train & Inference
### Train

The training logs will be saved to `work_dirs/[work_dir_name]`.

```bash
cd TopoFG
mkdir work_dirs

./tools/dist_train.sh 8 [work_dir_name] [--autoscale-lr]
```

### Evaluate
You can set `--show` to visualize the results.

```bash
./tools/dist_test.sh 8 [work_dir_name] [--show]
```


## Main Results
> The result is based on the `v2.1.0` OpenLane-V2 devkit and metrics. 
### Results on OpenLane-V2 subset-A val

We provide results on **[Openlane-V2](https://github.com/OpenDriveLab/OpenLane-V2) subset-A val** set.

### Results on subset_A

| Method | Epoch | OLS ↑ | DETₗ ↑ | DETₜ ↑ | TOPₗ ↑ | TOPₜ ↑ |
|--------|:-----:|-------|--------|--------|--------|--------|
| STSU | 24 | 29.3 | 12.7 | 43.0 | 2.9 | 19.8 |
| VectorMapNet | 24 | 24.9 | 11.1 | 41.7 | 2.7 | 9.2 |
| MapTR | 24 | 31.0 | 17.7 | 43.5 | 5.9 | 15.1 |
| TopoNet | 24 | 39.8 | 28.6 | 48.6 | 10.9 | 23.8 |
| TopoMLP | 24 | 44.1 | 28.5 | **49.5** | 21.7 | 26.9 |
| TopoLogic | 24 | 44.1 | 29.9 | 47.2 | 23.9 | 25.4 |
| **TopoFG (Ours)** | 24 | **48.0** | **33.8** | 47.2 | **30.8** | **30.9** |


## Weights
|    Method    | Backbone  | Epoch | Dataset | OLS |Version | Config | Download |  
| :----------: | :-------: | :---: | :-------------: | :--------------: | :-------------: | :--------------: | :------: |
| **TopoLogic**  | ResNet-50 |  24   |   subset-A | 48.0 | OpenLane-V2-v2.1.0 | [config](/projects/configs/topofg_r50_8x1_24e_olv2_subset_A.py) | [ckpt](https://drive.google.com/file/d/12lkFKd_sJ5kFQjfgXzdeLidWjPHK8fQ6/view?usp=drive_link) / [log](https://drive.google.com/file/d/1oCj2PlHsLjihcID7d-KCEmAj_ArNQqDi/view?usp=drive_link) |
## Acknowledgements
We thank these great works and open-source codebases:
- [TopoLogic](https://github.com/Franpin/TopoLogic)
- [TopoNet](https://github.com/OpenDriveLab/TopoNet)
- [Openlane-V2](https://github.com/OpenDriveLab/OpenLane-V2)
- [MapTR, MapTRv2](https://github.com/hustvl/MapTR)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)

## Citation
If you find our work is useful, please give this repo a star and cite our work as:
```bibtex
@article{xu2025fine,
  title={Fine-Grained Representation for Lane Topology Reasoning},
  author={Xu, Guoqing and Li, Yiheng and Yang, Yang},
  journal={arXiv preprint arXiv:2511.12590},
  year={2025}
}
```
