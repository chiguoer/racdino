#!/usr/bin/env python
"""调试 DINOv2 Adapter 的张量形状 (GPU 模式)"""

import torch
import sys

# 检查 CUDA
if not torch.cuda.is_available():
    print("⚠️  CUDA 不可用，MSDeformAttn 仅支持 GPU")
    print("请设置 CUDA_VISIBLE_DEVICES 并确保 GPU 可用")
    sys.exit(1)

device = torch.device("cuda")
print(f"使用设备: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

from models.backbones import DinoAdapter

def conv_output_size(input_size):
    """SPM 卷积输出尺寸公式: stride=2, padding=1, kernel=3"""
    return (input_size - 1) // 2 + 1

def debug_shapes():
    print("=" * 70)
    print("  DINOv2 Adapter 张量形状调试 (GPU 模式)")
    print("=" * 70)
    
    # 创建 adapter 并移动到 GPU
    adapter = DinoAdapter(
        num_heads=6,
        embed_dim=384,
        depth=12,
        pretrained_vit=False,
        freeze_dino=False,
        patch_size=14
    ).to(device)
    adapter.eval()
    
    # 测试 RaCFormer 默认尺寸: 256x704
    h, w = 256, 704
    print(f"\n测试尺寸: {h}x{w}")
    print("-" * 70)
    
    # 计算 padding 后的尺寸
    patch_size = 14
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    h_padded = h + pad_h
    w_padded = w + pad_w
    
    print(f"原始尺寸: {h}x{w}")
    print(f"Padding: h+{pad_h}, w+{pad_w}")
    print(f"Padding后: {h_padded}x{w_padded}")
    print()
    
    # 计算各个特征图的尺寸 (使用正确的公式)
    print("预期特征图尺寸:")
    print(f"  H_vit, W_vit = {h_padded // patch_size}, {w_padded // patch_size}")
    print(f"  H_adapt, W_adapt = {h_padded // 16}, {w_padded // 16}")
    print()
    
    # 使用正确的 SPM 输出尺寸计算
    # SPM stem: H/4, W/4
    H1 = h_padded // 4
    W1 = w_padded // 4
    # c2 = conv_output_size(c1)
    H2 = conv_output_size(H1)
    W2 = conv_output_size(W1)
    # c3 = conv_output_size(c2)
    H3 = conv_output_size(H2)
    W3 = conv_output_size(W2)
    # c4 = conv_output_size(c3)
    H4 = conv_output_size(H3)
    W4 = conv_output_size(W3)
    
    print("SPM 实际输出尺寸 (使用正确公式):")
    print(f"  c1 (stem): {H1}x{W1}")
    print(f"  c2: {H2}x{W2} (L2 = {H2*W2})")
    print(f"  c3: {H3}x{W3} (L3 = {H3*W3})")
    print(f"  c4: {H4}x{W4} (L4 = {H4*W4})")
    total_spm = H2*W2 + H3*W3 + H4*W4
    print(f"  总元素数: {total_spm}")
    print()
    
    print("ViT 输出尺寸:")
    H_vit = h_padded // 14
    W_vit = w_padded // 14
    print(f"  {H_vit}x{W_vit} (总元素: {H_vit * W_vit})")
    print()
    
    # 实际前向传播测试
    print("=" * 70)
    print("实际前向传播测试:")
    print("-" * 70)
    
    batch_size = 1
    img = torch.randn(batch_size, 3, h, w, device=device)
    
    try:
        with torch.no_grad():
            features, x_out = adapter(img)
        print("✅ 前向传播成功!")
        print(f"   输出特征: {len(features)}层")
        for i, feat in enumerate(features):
            print(f"   f{i+1}: {feat.shape}")
        print(f"   x_out: {x_out.shape}")
    except AssertionError as e:
        print(f"❌ 断言失败: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_shapes()


