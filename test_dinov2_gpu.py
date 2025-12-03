#!/usr/bin/env python
"""
DINOv2 Adapter GPU 测试脚本
这个脚本会在 GPU 上完整测试 DINOv2 Adapter 的功能
"""

import sys
import torch

print("=" * 70)
print("  DINOv2 Adapter GPU 完整测试")
print("=" * 70)

# 1. 检查 CUDA
print("\n[1] 检查 CUDA 环境...")
if not torch.cuda.is_available():
    print("❌ CUDA 不可用！MSDeformAttn 仅支持 GPU。")
    print("   请确保：")
    print("   1. 已安装 CUDA 版本的 PyTorch")
    print("   2. 设置了正确的 CUDA_VISIBLE_DEVICES")
    print("   3. GPU 驱动正常")
    sys.exit(1)

device = torch.device("cuda")
print(f"✅ CUDA 可用")
print(f"   GPU: {torch.cuda.get_device_name(0)}")
print(f"   CUDA 版本: {torch.version.cuda}")
print(f"   PyTorch 版本: {torch.__version__}")

# 2. 检查 MultiScaleDeformableAttention CUDA 扩展
print("\n[2] 检查 CUDA 扩展...")
try:
    import MultiScaleDeformableAttention as MSDA
    print("✅ MultiScaleDeformableAttention CUDA 扩展加载成功")
except ImportError as e:
    print(f"❌ CUDA 扩展加载失败: {e}")
    print("   请运行: cd models/backbones/nets/ops && python setup.py build_ext --inplace")
    sys.exit(1)

# 3. 导入 DinoAdapter
print("\n[3] 导入 DinoAdapter...")
try:
    from models.backbones import DinoAdapter
    print("✅ DinoAdapter 导入成功")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 创建模型并移动到 GPU
print("\n[4] 创建 DinoAdapter 并移动到 GPU...")
try:
    adapter = DinoAdapter(
        num_heads=6,  # ViT-Small
        embed_dim=384,
        depth=12,
        pretrained_vit=False,  # 跳过权重加载
        freeze_dino=False,
        patch_size=14
    )
    adapter = adapter.to(device)
    adapter.eval()
    print("✅ DinoAdapter 创建成功并已移动到 GPU")
    print(f"   embed_dim: {adapter.embed_dim}")
    print(f"   num_heads: {adapter.num_heads}")
    print(f"   patch_size: {adapter.patch_size}")
except Exception as e:
    print(f"❌ 创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. 测试前向传播
print("\n[5] 测试前向传播 (256x704)...")
try:
    h, w = 256, 704
    batch_size = 2
    
    # 创建输入张量（确保在 GPU 上）
    img = torch.randn(batch_size, 3, h, w, device=device, dtype=torch.float32)
    print(f"   输入张量: shape={img.shape}, device={img.device}")
    
    with torch.no_grad():
        features, x_out = adapter(img)
    
    print("✅ 前向传播成功！")
    print(f"   输出特征数量: {len(features)}")
    for i, f in enumerate(features):
        print(f"   f{i+1}: {f.shape}, device={f.device}")
    print(f"   x_out: {x_out.shape}, device={x_out.device}")
    
except Exception as e:
    print(f"❌ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. 测试多种尺寸
print("\n[6] 测试多种图像尺寸...")
test_sizes = [
    (224, 224),
    (256, 704),
    (480, 640),
    (300, 500),
]

all_passed = True
for h, w in test_sizes:
    try:
        img = torch.randn(1, 3, h, w, device=device)
        with torch.no_grad():
            features, x_out = adapter(img)
        print(f"   ✅ {h}x{w}: 成功 (f1={features[0].shape[2:]})")
    except Exception as e:
        print(f"   ❌ {h}x{w}: 失败 - {e}")
        all_passed = False

# 7. 总结
print("\n" + "=" * 70)
if all_passed:
    print("🎉 所有测试通过！DINOv2 Adapter 工作正常。")
    print("\n下一步：")
    print("  1. 运行完整检查: python tools/check_dinov2_integration.py")
    print("  2. 开始训练: bash tools/dist_train.sh configs/racformer_r50_nuimg_704x256_f8_with_dinov2_fixed.py 2")
else:
    print("⚠️  部分测试失败，请检查上述错误。")
print("=" * 70)

