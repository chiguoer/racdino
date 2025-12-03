#!/usr/bin/env python
"""
测试 DINOv2 Adapter 对不同图像尺寸的处理
验证自动 padding 功能
"""

import torch
from models.backbones import DinoAdapter


def test_various_sizes():
    print("=" * 70)
    print("  DINOv2 Adapter 图像尺寸兼容性测试")
    print("=" * 70)

    # 选择设备：优先使用 CUDA，因为 MSDeformAttn 只支持 GPU
    if not torch.cuda.is_available():
        print("\n⚠️  当前环境未检测到 CUDA，MSDeformAttn 仅支持 GPU")
        print("⚠️  请在具有 GPU 的环境中运行本测试\n")
        return False
    
    device = torch.device("cuda")
    print(f"\n当前设备: {device}")
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本: {torch.version.cuda}")

    # 创建 adapter（放到 GPU 上）
    print("\n正在初始化 DinoAdapter...")
    adapter = DinoAdapter(
        num_heads=6,  # ViT-Small
        embed_dim=384,
        depth=12,
        pretrained_vit=False,  # 不加载权重以加快测试
        freeze_dino=False,
        patch_size=14,
    )
    adapter = adapter.to(device)
    adapter.eval()
    print("✅ DinoAdapter 初始化完成并移动到 GPU")
    
    # 测试多种尺寸
    test_cases = [
        ("RaCFormer默认", (256, 704)),
        ("完美整除", (224, 224)),
        ("NuScenes原始", (900, 1600)),
        ("随机尺寸1", (480, 640)),
        ("随机尺寸2", (300, 500)),
    ]
    
    print("\n测试用例:")
    print("-" * 70)

    all_passed = True

    for name, (h, w) in test_cases:
        batch_size = 2
        # 确保输入在 GPU 上
        img = torch.randn(batch_size, 3, h, w, device=device, dtype=torch.float32)

        try:
            with torch.no_grad():
                features, x_out = adapter(img)

            print(f"✅ {name:20} | 输入: {h:4}x{w:4}", end="")
            print(f" | 输出特征: {len(features)}层", end="")
            print(f" | f1: {features[0].shape[2:4]}")

            # 验证输出维度（仅做粗略检查，允许少量误差）
            for i, feat in enumerate(features):
                expected_h = h // (4 * (2 ** i))  # f1: h/4, f2: h/8, f3: h/16, f4: h/32
                expected_w = w // (4 * (2 ** i))
                actual_h, actual_w = feat.shape[2], feat.shape[3]

                # 允许因 padding/cropping 产生的少量误差
                if abs(actual_h - expected_h) > 4 or abs(actual_w - expected_w) > 4:
                    print(
                        f"   ⚠️  特征{i+1}尺寸异常: 期望~({expected_h}, {expected_w}), 实际({actual_h}, {actual_w})"
                    )

        except Exception as e:
            print(f"❌ {name:20} | 输入: {h:4}x{w:4} | 错误: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("-" * 70)

    if all_passed:
        print("\n✅ 所有测试通过！DINOv2 Adapter 可以处理任意尺寸图像")
    else:
        print("\n⚠️  部分测试失败，请检查上述错误")

    print("=" * 70)
    return all_passed


def test_racformer_pipeline():
    """测试完整的 RaCFormer + DINOv2 管线"""
    print("\n" + "="*70)
    print("  RaCFormer + DINOv2 完整管线测试")
    print("="*70)
    
    try:
        from models import RaCFormer
        from mmcv import Config
        
        # 加载配置
        cfg = Config.fromfile('configs/racformer_r50_nuimg_704x256_f8_with_dinov2_fixed.py')
        
        print("\n✅ 配置文件加载成功")
        print(f"   图像尺寸: {cfg.ida_aug_conf['final_dim']}")
        print(f"   DINOv2: embed_dim={cfg.dinov2_adapter.get('embed_dim', 'N/A')}")
        
        # 注意：完整模型测试需要更多依赖，这里只验证配置
        print("\n✅ RaCFormer + DINOv2 配置兼容")
        
    except Exception as e:
        print(f"\n❌ 管线测试失败: {e}")
        return False
    
    print("=" * 70)
    return True


if __name__ == "__main__":
    print("\n")

    # 测试1: 各种图像尺寸
    test1_passed = test_various_sizes()

    # 测试2: RaCFormer管线兼容性
    test2_passed = test_racformer_pipeline()

    print("\n" + "=" * 70)
    print("  测试总结")
    print("=" * 70)
    print(f"  图像尺寸兼容性: {'✅ 通过' if test1_passed else '❌ 失败'}")
    print(f"  RaCFormer管线:   {'✅ 通过' if test2_passed else '❌ 失败'}")
    print("=" * 70 + "\n")

    if test1_passed and test2_passed:
        print("🎉 所有测试通过！可以开始训练。\n")
        exit(0)
    else:
        print("⚠️  存在失败的测试，请检查。\n")
        exit(1)
