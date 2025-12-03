#!/usr/bin/env python
"""详细调试 DINOv2 Adapter 的张量形状"""

import torch
import sys
sys.path.insert(0, '.')

# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

if device.type != "cuda":
    print("⚠️  警告: MSDeformAttn 仅支持 CUDA，CPU 模式下无法完整测试")
    print("请设置 CUDA_VISIBLE_DEVICES 并确保 GPU 可用")
    sys.exit(1)

# 临时修改 adapter_modules.py 来打印调试信息
from models.backbones.nets.dino_v2_with_adapter.dino_v2_adapter import adapter_modules

# Monkey patch Injector.forward 来打印形状
original_injector_forward = adapter_modules.Injector.forward

def debug_injector_forward(self, query, reference_points, feat, spatial_shapes, level_start_index):
    print("\n" + "="*70)
    print("Injector.forward 调试信息:")
    print("-"*70)
    print(f"query.shape: {query.shape}, device: {query.device}")
    print(f"feat.shape: {feat.shape}, device: {feat.device}")
    print(f"reference_points.shape: {reference_points.shape}, device: {reference_points.device}")
    print(f"spatial_shapes: {spatial_shapes}, device: {spatial_shapes.device}")
    print(f"level_start_index: {level_start_index}, device: {level_start_index.device}")
    
    # 计算 spatial_shapes 定义的总元素数
    total_elements_expected = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item()
    actual_elements = feat.shape[1]
    
    print(f"\nspatial_shapes 定义的总元素数: {total_elements_expected}")
    print(f"feat 实际序列长度: {actual_elements}")
    print(f"匹配: {'✅' if total_elements_expected == actual_elements else '❌'}")
    print("="*70)
    
    return original_injector_forward(self, query, reference_points, feat, spatial_shapes, level_start_index)

adapter_modules.Injector.forward = debug_injector_forward

# 现在运行测试
from models.backbones import DinoAdapter

def test():
    print("="*70)
    print("  DINOv2 Adapter 详细形状调试 (GPU 模式)")
    print("="*70)
    
    # 创建模型并移动到 GPU
    adapter = DinoAdapter(
        num_heads=6,
        embed_dim=384,
        depth=12,
        pretrained_vit=False,
        freeze_dino=False,
        patch_size=14
    ).to(device)
    adapter.eval()
    
    h, w = 256, 704
    print(f"\n测试尺寸: {h}x{w}")
    
    batch_size = 1
    # 创建输入并移动到 GPU
    img = torch.randn(batch_size, 3, h, w, device=device)
    
    try:
        with torch.no_grad():
            features, x_out = adapter(img)
        print("\n✅ 前向传播成功!")
        print(f"输出特征数量: {len(features)}")
        for i, f in enumerate(features):
            print(f"  f{i+1}: {f.shape}")
        print(f"x_out: {x_out.shape}")
    except Exception as e:
        print(f"\n❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()

