#!/usr/bin/env python
"""
DINOv2 集成一键检查脚本
快速验证 DINOv2 模块是否正确集成并可以使用
"""

import sys
import os

def print_header(title):
    """打印标题"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_status(status, message):
    """打印状态信息"""
    icon = "✅" if status else "❌"
    print(f"{icon} {message}")

def check_import():
    """检查 1: DinoAdapter 导入"""
    print_header("检查 1: DinoAdapter 模块导入")
    try:
        from models.backbones import DinoAdapter
        print_status(True, "DinoAdapter 导入成功")
        print(f"   模块路径: {DinoAdapter.__module__}")
        return True
    except Exception as e:
        print_status(False, f"DinoAdapter 导入失败: {e}")
        print("\n   解决方法:")
        print("   检查 models/backbones/nets/dino_v2_with_adapter/__init__.py")
        print("   确保导入语句为: from .dino_v2_adapter import DinoAdapter")
        return False

def check_config():
    """检查 2: 配置文件"""
    print_header("检查 2: 配置文件")
    
    config_files = [
        ("基线配置", "configs/racformer_r50_nuimg_704x256_f8.py", True),
        ("修正配置", "configs/racformer_r50_nuimg_704x256_f8_with_dinov2_fixed.py", True),
        ("原始配置", "configs/racformer_r50_nuimg_704x256_f8_with_dinov2.py", False),
    ]
    
    all_ok = True
    for name, path, should_exist in config_files:
        exists = os.path.exists(path)
        if should_exist:
            print_status(exists, f"{name}: {path}")
            if not exists:
                all_ok = False
        else:
            status_text = "存在（不推荐使用）" if exists else "不存在"
            print(f"   {name}: {status_text}")
    
    # 尝试加载修正配置
    fixed_config = "configs/racformer_r50_nuimg_704x256_f8_with_dinov2_fixed.py"
    if os.path.exists(fixed_config):
        try:
            from mmcv import Config
            cfg = Config.fromfile(fixed_config)
            print_status(True, "修正配置文件加载成功")
            
            # 检查 DINOv2 配置
            if hasattr(cfg, 'dinov2_adapter') and 'dinov2_adapter' in cfg.model:
                print_status(True, "DINOv2 配置正确")
            else:
                print_status(False, "DINOv2 配置缺失")
                all_ok = False
        except Exception as e:
            print_status(False, f"配置文件加载失败: {e}")
            all_ok = False
    
    return all_ok

def check_weights():
    """检查 3: 预训练权重"""
    print_header("检查 3: 预训练权重")
    
    weight_paths = [
        ("ResNet50 COCO", "pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth", True),
        ("DINOv2 ViT-Base", "weight/dinov2_vitb14_pretrain.pth", False),
        ("DINOv2 ViT-Base (备选)", "pretrain/dinov2_vitb14_pretrain.pth", False),
        ("DINOv2 ViT-Small", "weight/dinov2_vits14_pretrain.pth", False),
    ]
    
    resnet_ok = False
    dinov2_ok = False
    
    for name, path, required in weight_paths:
        exists = os.path.exists(path)
        if required:
            print_status(exists, f"{name}: {path}")
            if exists and "ResNet" in name:
                resnet_ok = True
        else:
            if exists:
                print_status(True, f"{name}: {path} (找到)")
                if "DINOv2" in name:
                    dinov2_ok = True
            else:
                print(f"   {name}: 未找到 (可选，将从 torch.hub 下载)")
    
    if not resnet_ok:
        print("\n   ⚠️  警告: 未找到 ResNet50 预训练权重")
        print("   下载地址: https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth")
    
    if not dinov2_ok:
        print("\n   ℹ️  信息: 未找到本地 DINOv2 权重")
        print("   首次运行时会自动从 torch.hub 下载")
    
    return resnet_ok

def check_cuda_extensions():
    """检查 4: CUDA 扩展"""
    print_header("检查 4: CUDA 扩展")
    
    cuda_ext_path = "models/csrc/build"
    if os.path.exists(cuda_ext_path):
        print_status(True, f"CUDA 扩展已编译: {cuda_ext_path}")
        return True
    else:
        print_status(False, "CUDA 扩展未编译")
        print("\n   编译方法:")
        print("   cd models/csrc")
        print("   python setup.py build_ext --inplace")
        return False

def check_dataset():
    """检查 5: 数据集"""
    print_header("检查 5: 数据集 (可选)")
    
    dataset_paths = [
        "/data/dataset/RacFormer/nuscenes/",
        "data/nuscenes/",
    ]
    
    dataset_found = False
    for path in dataset_paths:
        if os.path.exists(path):
            print_status(True, f"数据集路径: {path}")
            
            # 检查必要文件
            required_files = [
                "nuscenes_infos_train_sweep.pkl",
                "nuscenes_infos_val_sweep.pkl",
            ]
            
            all_files_exist = True
            for file in required_files:
                file_path = os.path.join(path, file)
                exists = os.path.exists(file_path)
                print(f"   {'✅' if exists else '❌'} {file}")
                if not exists:
                    all_files_exist = False
            
            dataset_found = True
            return all_files_exist
    
    print_status(False, "未找到数据集")
    print("\n   请下载 nuScenes 数据集并放置在以下路径之一:")
    for path in dataset_paths:
        print(f"   - {path}")
    
    return False

def print_summary(results):
    """打印总结"""
    print_header("检查总结")
    
    checks = [
        ("DinoAdapter 导入", results['import']),
        ("配置文件", results['config']),
        ("ResNet 预训练权重", results['weights']),
        ("CUDA 扩展", results['cuda']),
        ("数据集", results['dataset']),
    ]
    
    all_critical_ok = results['import'] and results['config'] and results['cuda']
    
    for name, status in checks:
        print_status(status, name)
    
    print("\n" + "-"*70)
    if all_critical_ok:
        print("✅ 核心组件检查通过！可以开始训练。")
        print("\n推荐使用以下命令训练:")
        print("torchrun --nproc_per_node 8 train.py --config configs/racformer_r50_nuimg_704x256_f8_with_dinov2_fixed.py")
    else:
        print("❌ 存在关键问题，请先解决上述错误。")
        print("\n详细修复方法请参考:")
        print("- 配置文件修正说明.md")
        print("- RACFORMER_DINOV2_INTEGRATION_REPORT.md")
    
    if not results['weights']:
        print("\n⚠️  警告: 缺少 ResNet 预训练权重可能影响训练效果。")
    
    if not results['dataset']:
        print("\n⚠️  警告: 未找到数据集，需要准备数据才能训练。")
    
    print("="*70 + "\n")

def main():
    """主函数"""
    print("\n" + "="*70)
    print("  RaCFormer + DINOv2 集成检查工具")
    print("  一键验证所有组件是否正确配置")
    print("="*70)
    
    results = {
        'import': check_import(),
        'config': check_config(),
        'weights': check_weights(),
        'cuda': check_cuda_extensions(),
        'dataset': check_dataset(),
    }
    
    print_summary(results)
    
    # 返回退出码
    all_critical_ok = results['import'] and results['config'] and results['cuda']
    sys.exit(0 if all_critical_ok else 1)

if __name__ == "__main__":
    main()

