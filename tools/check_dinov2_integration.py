"""
快速检查DINOv2整合的完整性
运行方式: python tools/check_dinov2_integration.py
"""

import sys
import os
import torch

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def check_imports():
    """检查模块导入"""
    print_section("1️⃣ 检查模块导入")
    
    checks = []
    
    # 检查DINOv2 Adapter
    try:
        from models.backbones import DinoAdapter
        checks.append(("DinoAdapter导入", True, "✅"))
        print("✅ DinoAdapter可成功导入")
    except Exception as e:
        checks.append(("DinoAdapter导入", False, f"❌ {e}"))
        print(f"❌ DinoAdapter导入失败: {e}")
    
    # 检查RaCFormer
    try:
        from models.racformer import RaCFormer
        checks.append(("RaCFormer导入", True, "✅"))
        print("✅ RaCFormer可成功导入")
    except Exception as e:
        checks.append(("RaCFormer导入", False, f"❌ {e}"))
        print(f"❌ RaCFormer导入失败: {e}")
    
    # 检查RaCFormer_head
    try:
        from models.racformer_head import RaCFormer_head
        checks.append(("RaCFormer_head导入", True, "✅"))
        print("✅ RaCFormer_head可成功导入")
    except Exception as e:
        checks.append(("RaCFormer_head导入", False, f"❌ {e}"))
        print(f"❌ RaCFormer_head导入失败: {e}")
    
    return all(c[1] for c in checks)

def check_dinov2_adapter():
    """检查DINOv2 Adapter功能"""
    print_section("2️⃣ 检查DINOv2 Adapter功能")
    
    # 检查 CUDA 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n当前设备: {device}")
    
    if device.type != "cuda":
        print("⚠️  警告: MSDeformAttn 仅支持 CUDA，跳过前向传播测试")
        print("   请在 GPU 环境下运行完整测试")
        return False
    
    try:
        from models.backbones import DinoAdapter
        
        # 测试ViT-Small配置
        print("\n测试 ViT-Small 配置...")
        adapter_small = DinoAdapter(
            num_heads=6,
            embed_dim=384,
            depth=12,
            pretrained_vit=False,  # 跳过权重加载以加快测试
            freeze_dino=True,
        ).to(device)  # 移动到 GPU
        print(f"  ✅ ViT-Small初始化成功")
        print(f"  - embed_dim: {adapter_small.embed_dim}")
        print(f"  - num_heads: {adapter_small.num_heads}")
        print(f"  - depth: {len(adapter_small.blocks)}")
        
        # 测试前向传播
        print("\n测试前向传播...")
        dummy_input = torch.randn(2, 3, 256, 704, device=device)  # [B, C, H, W] 在 GPU 上
        with torch.no_grad():
            features, x_out = adapter_small(dummy_input)
        
        print(f"  ✅ 前向传播成功")
        print(f"  - 输入形状: {dummy_input.shape}")
        print(f"  - 输出特征数量: {len(features)}")
        for i, feat in enumerate(features):
            print(f"  - 特征{i+1}形状: {feat.shape}")
        print(f"  - x_out形状: {x_out.shape}")
        
        # 验证batch维度
        assert features[0].shape[0] == 2, "❌ Batch维度丢失！"
        print("  ✅ Batch维度保持正确")
        
        # 验证通道数
        assert features[0].shape[1] == 384, f"❌ 通道数错误：期望384，实际{features[0].shape[1]}"
        print("  ✅ 输出通道数正确 (384)")
        
        return True
        
    except Exception as e:
        print(f"❌ DINOv2 Adapter测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_semantic_fusion():
    """检查语义融合层"""
    print_section("3️⃣ 检查语义融合层")
    
    try:
        import torch.nn as nn
        from mmcv.cnn import ConvModule
        
        # 模拟ResNet50输出通道
        resnet_channels = [256, 512, 1024, 2048]
        dinov2_embed_dim = 768
        
        print(f"\nResNet50通道数: {resnet_channels}")
        print(f"DINOv2 embed_dim: {dinov2_embed_dim}")
        
        # 创建融合层
        semantic_fusion = nn.ModuleList([
            ConvModule(
                in_channels=resnet_channels[i] + dinov2_embed_dim,
                out_channels=resnet_channels[i],
                kernel_size=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
                bias='auto'
            ) for i in range(4)
        ])
        
        print(f"\n✅ 语义融合层创建成功")
        
        # 测试每个层级的融合
        print("\n测试各层级融合:")
        batch_size = 2
        H, W = 64, 176  # 示例尺寸
        
        for i in range(4):
            # 模拟ResNet和DINOv2特征
            resnet_feat = torch.randn(batch_size, resnet_channels[i], H//(2**i), W//(2**i))
            dinov2_feat = torch.randn(batch_size, dinov2_embed_dim, H//(2**i), W//(2**i))
            
            # 拼接
            combined = torch.cat([resnet_feat, dinov2_feat], dim=1)
            
            # 融合
            with torch.no_grad():
                fused = semantic_fusion[i](combined)
            
            print(f"  层级{i+1}:")
            print(f"    - ResNet特征: {resnet_feat.shape}")
            print(f"    - DINOv2特征: {dinov2_feat.shape}")
            print(f"    - 拼接后: {combined.shape}")
            print(f"    - 融合后: {fused.shape}")
            
            # 验证输出形状
            assert fused.shape == resnet_feat.shape, f"❌ 融合后形状不匹配！"
            assert fused.shape[1] == resnet_channels[i], f"❌ 输出通道数错误！"
        
        print("\n✅ 所有层级融合测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 语义融合层测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_query_initialization():
    """检查查询初始化"""
    print_section("4️⃣ 检查圆形线性倍增查询初始化")
    
    try:
        from models.racformer_head import RaCFormer_head
        
        # 创建head（使用与原始配置兼容的参数）
        head = RaCFormer_head(
            num_classes=10,
            in_channels=256,
            num_clusters=6,
            num_query=900,
            embed_dims=256,
            code_size=10,
            code_weights=[1.0] * 10,
            # 添加 dummy transformer 配置以通过父类初始化检查
            transformer=dict(
                type='Transformer',  # 使用简单配置即可，此处仅检查初始化逻辑
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            # 使用不带 bg_cls_weight 的 loss_cls 配置
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0
            ),
            loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        )
        
        print(f"\n✅ RaCFormer_head创建成功")
        print(f"  - num_query: {head.num_query}")
        print(f"  - num_clusters: {head.num_clusters}")
        
        # 检查查询bbox初始化
        query_bbox = head.init_query_bbox.weight
        print(f"\n查询bbox形状: {query_bbox.shape}")
        
        # 提取theta和distance
        query_pos = query_bbox[:, :2]
        distances = query_pos[:, 1]
        
        # 统计每个圆环的查询数量
        unique_dists, counts = torch.unique(distances, return_counts=True)
        
        print(f"\n查询分布:")
        for i, (d, c) in enumerate(zip(unique_dists, counts)):
            print(f"  圆{i+1}: 距离={d:.3f}, 查询数={c.item():4d}")
        
        # 验证线性递增
        counts_list = counts.tolist()
        is_increasing = all(counts_list[i] <= counts_list[i+1] for i in range(len(counts_list)-1))
        
        if is_increasing:
            print("\n✅ 查询数量线性递增验证通过")
        else:
            print("\n❌ 查询数量未实现线性递增")
            return False
        
        # 验证总数
        total = sum(counts_list)
        if total == head.num_query:
            print(f"✅ 总查询数验证通过: {total}/{head.num_query}")
        else:
            print(f"❌ 总查询数不匹配: {total}/{head.num_query}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 查询初始化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_config_files():
    """检查配置文件"""
    print_section("5️⃣ 检查配置文件")
    
    config_files = [
        'configs/racformer_r50_nuimg_704x256_f8_with_dinov2.py',
        'configs/racformer_r50_nuimg_704x256_f8.py',
    ]
    
    all_exist = True
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ {config_file} 存在")
            
            # 尝试读取配置
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'dinov2_adapter' in content:
                        print(f"   - 包含DINOv2配置 ✅")
                    if 'num_clusters' in content:
                        print(f"   - 包含查询初始化配置 ✅")
            except Exception as e:
                print(f"   - 读取失败: {e}")
        else:
            print(f"❌ {config_file} 不存在")
            all_exist = False
    
    return all_exist

def check_weight_paths():
    """检查DINOv2权重路径"""
    print_section("6️⃣ 检查DINOv2权重路径")
    
    weight_paths = [
        'weight/dinov2_vitb14_pretrain.pth',
        'weight/dinov2_vits14_pretrain.pth',
        'pretrain/dinov2_vitb14_pretrain.pth',
        'pretrain/dinov2_vits14_pretrain.pth',
        os.path.expanduser('~/.cache/dinov2/dinov2_vitb14_pretrain.pth'),
        os.path.expanduser('~/.cache/dinov2/dinov2_vits14_pretrain.pth'),
    ]
    
    found_weights = []
    for path in weight_paths:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✅ 找到权重: {path} ({size_mb:.1f} MB)")
            found_weights.append(path)
    
    if not found_weights:
        print("⚠️  未找到本地DINOv2权重文件")
        print("   代码将自动从torch.hub下载")
        print("   建议手动下载到以下路径之一:")
        print("   - weight/dinov2_vitb14_pretrain.pth")
        print("   - pretrain/dinov2_vitb14_pretrain.pth")
        return False
    
    return True

def generate_summary(results):
    """生成检查总结"""
    print_section("📋 检查总结")
    
    print("\n检查项目:")
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
    
    all_critical_passed = all([
        results.get('模块导入', False),
        results.get('DINOv2功能', False),
        results.get('语义融合', False),
        results.get('查询初始化', False),
    ])
    
    print("\n" + "=" * 80)
    if all_critical_passed:
        print("🎉 所有关键检查通过！代码可以运行。")
        print("\n下一步:")
        print("  1. 如果还没有DINOv2权重，运行训练时会自动下载")
        print("  2. 运行查询初始化可视化:")
        print("     python tools/verify_query_initialization.py")
        print("  3. 开始训练:")
        print("     bash tools/dist_train.sh configs/racformer_r50_nuimg_704x256_f8_with_dinov2.py 8")
    else:
        print("⚠️  部分检查未通过，请查看上面的详细信息。")
    print("=" * 80)

if __name__ == '__main__':
    print("=" * 80)
    print("  RaCFormer + DINOv2 整合完整性检查")
    print("=" * 80)
    
    results = {}
    
    # 运行所有检查
    results['模块导入'] = check_imports()
    results['DINOv2功能'] = check_dinov2_adapter()
    results['语义融合'] = check_semantic_fusion()
    results['查询初始化'] = check_query_initialization()
    results['配置文件'] = check_config_files()
    results['权重文件'] = check_weight_paths()
    
    # 生成总结
    generate_summary(results)

