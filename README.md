<<<<<<< HEAD
# racdino
radar; camera; dino; query init
=======
<div align="center">
<h1>RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion (CVPR 2025)</h1>

Xiaomeng Chu, Jiajun Deng, Guoliang You, Yifan Duan, Houqiang Li, Yanyong Zhang

<a href="https://arxiv.org/abs/2412.12725"><img src="https://img.shields.io/badge/arXiv-2412.12725-b31b1b" alt="arXiv"></a>
<a href="https://drive.google.com/file/d/10Ky3lQWC2MLkQCpY81Jz5yxd4xWF8tAq/view?usp=sharing" target="_blank"><img src="https://img.shields.io/badge/Checkpoint-Orange" alt="checkpoint"></a>
</div>

```bibtex
@inproceedings{chu2025racformer,
  title={RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion},
  author={Chu, Xiaomeng and Deng, Jiajun and You, Guoliang and Duan, Yifan and Li, Houqiang and Zhang, Yanyong},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={17081--17091},
  year={2025}
}
```

## Overview

This repository is an official implementation of [RaCFormer](https://openaccess.thecvf.com/content/CVPR2025/html/Chu_RaCFormer_Towards_High-Quality_3D_Object_Detection_via_Query-based_Radar-Camera_Fusion_CVPR_2025_paper.html), an innovative query-based 3D object detection method through cross-perspective radar-camera fusion.
<div style="text-align: center;">
    <img src="arch.jpg" alt="Dialogue_Teaser" width=100% >
</div>



## Environment

Install PyTorch 2.0 + CUDA 11.8:

```
conda create -n racformer python=3.8
conda activate racformer
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```


Install other dependencies:

```
pip install openmim
mim install mmcv-full==1.6.0
mim install mmdet==2.28.2
mim install mmsegmentation==0.30.0
mim install mmdet3d==1.0.0rc6
pip install setuptools==59.5.0
pip install numpy==1.23.5
```

Install turbojpeg and pillow-simd to speed up data loading (optional but important):

```
sudo apt-get update
sudo apt-get install -y libturbojpeg
pip install pyturbojpeg
pip uninstall pillow
pip install pillow-simd==9.0.0.post1
```

Compile CUDA extensions:

```
cd models/csrc
python setup.py build_ext --inplace
```

## Prepare Dataset

1. Download nuScenes from [https://www.nuscenes.org/nuscenes](https://www.nuscenes.org/nuscenes) and put it in `data/nuscenes`.
2. Download the generated info file from [Google Drive](https://drive.google.com/drive/folders/1Tec0I7tgJKF-w1_vVAScJ0wPek2YT28u?usp=sharing).
3. Folder structure:

```
data/nuscenes
├── maps
├── nuscenes_infos_test_sweep.pkl
├── nuscenes_infos_train_sweep.pkl
├── samples
├── sweeps
├── v1.0-test
└── v1.0-trainval
```

## Training

Download [pretrained ResNet-50](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth) and put it in directory `pretrain/`:

```
pretrain
├── cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth
```

Train RaCFormer with 8 GPUs:

```
torchrun --nproc_per_node 8 train.py --config configs/racformer_r50_nuimg_704x256_f8.py
```

## Evaluation

Download the [model weights](https://drive.google.com/file/d/10Ky3lQWC2MLkQCpY81Jz5yxd4xWF8tAq/view?usp=sharing).

Single-GPU evaluation:

```
export CUDA_VISIBLE_DEVICES=0
python val.py --config configs/racformer_r50_nuimg_704x256_f8.py --weights checkpoints/r50_nuimg_704x256.pth
```

Multi-GPU evaluation:

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node 8 val.py --config configs/racformer_r50_nuimg_704x256_f8.py --weights checkpoints/r50_nuimg_704x256.pth
```

## DINOv2 Integration (实验性功能)

本仓库已集成 DINOv2 语义增强模块，用于提升图像特征的语义表达能力。

### ⚠️ 重要提示

在使用 DINOv2 前，**必须先编译 CUDA 扩展**。请按以下步骤操作：

#### 方式 1: 使用一键修复脚本（推荐）

```bash
chmod +x fix_dinov2_import.sh
./fix_dinov2_import.sh
```

#### 方式 2: 手动编译

```bash
# 1. 编译 MultiScaleDeformableAttention CUDA 扩展
cd models/backbones/nets/ops
python setup.py build install
cd ../../../../

# 2. 验证安装
python -c "import MultiScaleDeformableAttention; print('✅ CUDA扩展安装成功')"

# 3. 测试 DinoAdapter 导入
python -c "from models.backbones import DinoAdapter; print('✅ DinoAdapter可导入')"
```

### 快速开始

1. **编译并验证 DINOv2 模块**
```bash
# 一键修复（推荐）
./fix_dinov2_import.sh

# 或运行完整检查
python check_dinov2_setup.py
```

2. **准备 DINOv2 预训练权重**

DINOv2 权重会自动从以下路径加载（优先级从高到低）：
- `weight/dinov2_vitb14_pretrain.pth` （ViT-Base）或 `weight/dinov2_vits14_pretrain.pth` （ViT-Small）
- `pretrain/dinov2_vitb14_pretrain.pth` 或 `pretrain/dinov2_vits14_pretrain.pth`
- `~/.cache/dinov2/`
- 自动从 `torch.hub` 下载

3. **使用修正后的配置文件训练**
```bash
# 推荐使用修正版配置（仅添加 DINOv2，其余与基线一致）
torchrun --nproc_per_node 8 train.py --config configs/racformer_r50_nuimg_704x256_f8_with_dinov2_fixed.py
```

### 配置文件说明

- ✅ **推荐**: `configs/racformer_r50_nuimg_704x256_f8_with_dinov2_fixed.py`
  - 基于原始配置的**最小修改版本**
  - 仅添加 DINOv2 语义增强模块
  - 其余配置（数据管线、优化器、训练策略）与基线完全一致
  
- ⚠️ **不推荐**: `configs/racformer_r50_nuimg_704x256_f8_with_dinov2.py`
  - 混入了大量与 DINOv2 无关的改动（数据管线、雷达编码器、训练超参等）
  - 可能导致配置冲突和运行错误
  - 建议使用上述修正版

### DINOv2 集成位置

DINOv2 adapter 位于 **ResNet 编码后、深度估计之前**：
```
输入图像 → ResNet编码 → DINOv2语义增强 → FPN → 深度估计 → BEV变换
```

这一位置设计基于以下考虑：
1. **语义理解优先**: DINOv2 增强的语义特征有助于后续的深度估计
2. **特征融合**: ResNet 和 DINOv2 特征在通道维度拼接后，通过 1×1 卷积融合
3. **论文依据**: RaCFormer 论文指出"高质量的图像表征是准确深度估计的关键"

### 验证工具

运行验证脚本检查配置正确性：
```bash
python tools/verify_dinov2_config.py
```

该脚本会检查：
- DinoAdapter 模块导入
- 配置文件加载
- DINOv2 配置参数
- 与基线配置的一致性
- 原始配置与修正配置的对比

详细信息请参考：[配置文件修正说明.md](配置文件修正说明.md)

### 已修复的问题

1. ✅ **DinoAdapter 导入错误**
   - 错误: `ModuleNotFoundError: No module named 'dino_v2_adapter'`
   - 修复: 更正了 `models/backbones/nets/dino_v2_with_adapter/__init__.py` 中的导入语句

2. ✅ **配置文件混乱**
   - 问题: 原配置文件混入了过多与 DINOv2 无关的改动
   - 修复: 创建了基于基线的最小修改版本

3. ✅ **特征维度匹配**
   - 问题: ResNet 和 DINOv2 输出维度不一致
   - 修复: 动态推断 ResNet 输出通道，自动调整融合层

### 相关文档

- [RACFORMER_DINOV2_INTEGRATION_REPORT.md](RACFORMER_DINOV2_INTEGRATION_REPORT.md) - 完整的集成报告
- [配置文件修正说明.md](配置文件修正说明.md) - 配置文件问题分析与修正
- [DIMENSION_CHECK.md](DIMENSION_CHECK.md) - 特征维度匹配验证

## Acknowledgements

Many thanks to these excellent open-source projects:

* 3D Detection: [SparseBEV](https://github.com/MCG-NJU/SparseBEV), [PETR v2](https://github.com/megvii-research/PETR), [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [BEVDet](https://github.com/HuangJunJie2017/BEVDet) 
* Codebase: [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
* DINOv2: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
>>>>>>> b151592 (加入dino语义增强器的racfogmer代码)
