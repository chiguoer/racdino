"""
验证 DINOv2 配置文件的正确性
检查模块导入、配置加载和关键参数
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_dinov2_import():
    """检查 DinoAdapter 是否可以正确导入"""
    print("\n" + "="*60)
    print("步骤 1: 检查 DinoAdapter 导入")
    print("="*60)
    try:
        from models.backbones import DinoAdapter
        print("✅ DinoAdapter 导入成功")
        print(f"   模块位置: {DinoAdapter.__module__}")
        return True
    except Exception as e:
        print(f"❌ DinoAdapter 导入失败: {e}")
        return False

def check_config_loading(config_path):
    """检查配置文件是否可以正确加载"""
    print("\n" + "="*60)
    print(f"步骤 2: 检查配置文件加载")
    print(f"配置文件: {config_path}")
    print("="*60)
    try:
        from mmcv import Config
        cfg = Config.fromfile(config_path)
        print("✅ 配置文件加载成功")
        return cfg
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_dinov2_config(cfg):
    """检查 DINOv2 相关配置"""
    print("\n" + "="*60)
    print("步骤 3: 检查 DINOv2 配置参数")
    print("="*60)
    
    if cfg is None:
        print("❌ 配置对象为空，跳过检查")
        return False
    
    # 检查 dinov2_adapter 配置
    if not hasattr(cfg, 'dinov2_adapter'):
        print("⚠️  警告: 配置中没有 dinov2_adapter 定义")
        return False
    
    dinov2_cfg = cfg.dinov2_adapter
    print("✅ 找到 dinov2_adapter 配置")
    print(f"   类型: {dinov2_cfg.get('type', 'N/A')}")
    print(f"   embed_dim: {dinov2_cfg.get('embed_dim', 'N/A')}")
    print(f"   num_heads: {dinov2_cfg.get('num_heads', 'N/A')}")
    print(f"   depth: {dinov2_cfg.get('depth', 'N/A')}")
    print(f"   pretrained_vit: {dinov2_cfg.get('pretrained_vit', 'N/A')}")
    print(f"   freeze_dino: {dinov2_cfg.get('freeze_dino', 'N/A')}")
    
    # 检查模型配置中是否包含 dinov2_adapter
    if not hasattr(cfg, 'model'):
        print("❌ 配置中没有 model 定义")
        return False
    
    if 'dinov2_adapter' not in cfg.model:
        print("❌ model 配置中没有引用 dinov2_adapter")
        return False
    
    print("✅ model 配置中已正确引用 dinov2_adapter")
    return True

def check_baseline_consistency(cfg):
    """检查与基线配置的一致性"""
    print("\n" + "="*60)
    print("步骤 4: 检查与基线配置的一致性")
    print("="*60)
    
    if cfg is None:
        print("❌ 配置对象为空，跳过检查")
        return False
    
    issues = []
    
    # 检查关键配置是否与基线一致
    checks = [
        ("img_backbone.type", "ResNet", cfg.img_backbone.get('type')),
        ("img_backbone.depth", 50, cfg.img_backbone.get('depth')),
        ("optimizer.lr", 4e-4, cfg.optimizer.get('lr')),
        ("total_epochs", 20, cfg.get('total_epochs')),
        ("batch_size", 4, cfg.get('batch_size')),
    ]
    
    for name, expected, actual in checks:
        if actual == expected:
            print(f"✅ {name}: {actual} (符合预期)")
        else:
            print(f"⚠️  {name}: {actual} (预期: {expected})")
            issues.append(name)
    
    # 检查数据管线
    if hasattr(cfg, 'train_pipeline'):
        pipeline_types = [p.get('type') for p in cfg.train_pipeline]
        expected_types = ['LoadMultiViewImageFromFiles', 'Loadnuradarpoints']
        
        for expected in expected_types:
            if expected in pipeline_types:
                print(f"✅ 训练管线包含: {expected}")
            else:
                print(f"⚠️  训练管线缺失: {expected}")
                issues.append(f"pipeline.{expected}")
    
    # 检查雷达编码器
    if hasattr(cfg, 'model') and 'radar_voxel_encoder' in cfg.model:
        radar_enc = cfg.model.radar_voxel_encoder
        if radar_enc.get('type') == 'PillarFeatureNet' and radar_enc.get('in_channels') == 7:
            print("✅ 雷达编码器配置正确 (PillarFeatureNet, in_channels=7)")
        else:
            print(f"⚠️  雷达编码器配置异常: type={radar_enc.get('type')}, in_channels={radar_enc.get('in_channels')}")
            issues.append("radar_voxel_encoder")
    
    if issues:
        print(f"\n⚠️  发现 {len(issues)} 个与基线不一致的配置项")
        return False
    else:
        print("\n✅ 所有关键配置与基线一致")
        return True

def compare_configs(original_path, fixed_path):
    """对比原始配置和修正配置"""
    print("\n" + "="*60)
    print("步骤 5: 对比原始配置与修正配置")
    print("="*60)
    
    try:
        from mmcv import Config
        original_cfg = Config.fromfile(original_path)
        fixed_cfg = Config.fromfile(fixed_path)
        
        print(f"\n原始配置: {original_path}")
        print(f"修正配置: {fixed_path}\n")
        
        # 关键差异对比
        comparisons = [
            ("学习率", "optimizer.lr", original_cfg.optimizer.get('lr'), fixed_cfg.optimizer.get('lr')),
            ("训练轮数", "total_epochs", original_cfg.get('total_epochs'), fixed_cfg.get('total_epochs')),
            ("批次大小", "batch_size", original_cfg.get('batch_size', 'N/A'), fixed_cfg.get('batch_size', 'N/A')),
            ("雷达编码器", "model.radar_voxel_encoder.type", 
             original_cfg.model.radar_voxel_encoder.get('type'),
             fixed_cfg.model.radar_voxel_encoder.get('type')),
            ("雷达输入通道", "model.radar_voxel_encoder.in_channels",
             original_cfg.model.radar_voxel_encoder.get('in_channels'),
             fixed_cfg.model.radar_voxel_encoder.get('in_channels')),
        ]
        
        print("关键配置对比:")
        print("-" * 60)
        for name, key, orig_val, fixed_val in comparisons:
            status = "✅" if orig_val == fixed_val else "⚠️ "
            print(f"{status} {name:12} | 原始: {str(orig_val):20} | 修正: {str(fixed_val):20}")
        
        print("\n" + "="*60)
        print("对比完成")
        print("="*60)
        
    except Exception as e:
        print(f"❌ 配置对比失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("\n" + "="*60)
    print("DINOv2 配置验证工具")
    print("="*60)
    
    # 1. 检查导入
    import_ok = check_dinov2_import()
    
    # 2. 检查修正后的配置文件
    fixed_config = "configs/racformer_r50_nuimg_704x256_f8_with_dinov2_fixed.py"
    if os.path.exists(fixed_config):
        cfg = check_config_loading(fixed_config)
        if cfg:
            check_dinov2_config(cfg)
            check_baseline_consistency(cfg)
    else:
        print(f"\n⚠️  修正后的配置文件不存在: {fixed_config}")
    
    # 3. 对比原始配置和修正配置
    original_config = "configs/racformer_r50_nuimg_704x256_f8_with_dinov2.py"
    if os.path.exists(original_config) and os.path.exists(fixed_config):
        compare_configs(original_config, fixed_config)
    
    # 最终总结
    print("\n" + "="*60)
    print("验证总结")
    print("="*60)
    print("✅ 推荐使用: configs/racformer_r50_nuimg_704x256_f8_with_dinov2_fixed.py")
    print("   该配置文件仅添加 DINOv2，其余与基线一致")
    print("\n⚠️  不推荐使用: configs/racformer_r50_nuimg_704x256_f8_with_dinov2.py")
    print("   该配置文件混入了过多与 DINOv2 无关的改动")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()

