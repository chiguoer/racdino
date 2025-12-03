# DINOv2适配器维度匹配检查文档

## 输入输出维度说明

### 1. 输入维度

**图像输入** (`img`):
- 形状: `[BNT, C, H, W]`
- 其中:
  - `B` = Batch size
  - `N` = Num_frames (通常为8)
  - `T` = Num_cameras (通常为6)
  - `C` = 3 (RGB通道)
  - `H, W` = 图像高度和宽度 (例如: 256, 704)

### 2. ResNet50骨干网络输出

**ResNet50输出4个特征图**:
- `feat0`: `[BNT, 256, H/4, W/4]`  (stage 1)
- `feat1`: `[BNT, 512, H/8, W/8]`  (stage 2)
- `feat2`: `[BNT, 1024, H/16, W/16]` (stage 3)
- `feat3`: `[BNT, 2048, H/32, W/32]` (stage 4)

### 3. DINOv2适配器输出

**DINOv2适配器输出4个特征图**:
- `f1`: `[BNT, embed_dim, H_f1, W_f1]` (embed_dim=384 for VIT-S, 768 for VIT-B)
- `f2`: `[BNT, embed_dim, H_f2, W_f2]`
- `f3`: `[BNT, embed_dim, H_f3, W_f3]`
- `f4`: `[BNT, embed_dim, H_f4, W_f4]`

**注意**: DINOv2适配器的空间尺寸取决于输入图像尺寸和patch_size(14)

### 4. 特征融合层

**融合层配置** (4个融合层，对应4个特征层级):

```python
semantic_fusion[0]: Conv(256 + 384 → 256)  # 对应ResNet stage 1
semantic_fusion[1]: Conv(512 + 384 → 512)  # 对应ResNet stage 2
semantic_fusion[2]: Conv(1024 + 384 → 1024) # 对应ResNet stage 3
semantic_fusion[3]: Conv(2048 + 384 → 2048) # 对应ResNet stage 4
```

### 5. 维度匹配流程

```
输入图像: [BNT, 3, H, W]
    ↓
ResNet50: [BNT, 256, H/4, W/4], [BNT, 512, H/8, W/8], [BNT, 1024, H/16, W/16], [BNT, 2048, H/32, W/32]
    ↓
DINOv2适配器: [BNT, 384, H_d1, W_d1], [BNT, 384, H_d2, W_d2], [BNT, 384, H_d3, W_d3], [BNT, 384, H_d4, W_d4]
    ↓
空间插值: 调整DINOv2特征的空间尺寸以匹配ResNet特征
    ↓
特征拼接: [BNT, 256+384, H/4, W/4], [BNT, 512+384, H/8, W/8], ...
    ↓
融合卷积: [BNT, 256, H/4, W/4], [BNT, 512, H/8, W/8], ...
```

## 维度验证检查点

### 检查点1: 特征数量匹配
- ✅ ResNet输出4个特征图
- ✅ DINOv2输出4个特征图
- ✅ 数量匹配

### 检查点2: Batch维度匹配
- ✅ ResNet特征: `[BNT, C, H, W]`
- ✅ DINOv2特征: `[BNT, embed_dim, H, W]`
- ✅ Batch维度匹配

### 检查点3: 通道维度匹配
- ✅ 融合层输入通道 = ResNet通道 + DINOv2通道
- ✅ 融合层输出通道 = ResNet通道
- ✅ 通道维度匹配

### 检查点4: 空间维度匹配
- ⚠️ 空间维度通过插值自动匹配
- ✅ 使用双线性插值调整DINOv2特征尺寸

## 权重加载路径

优先级顺序:
1. `weight/dinov2_vits14_pretrain.pth` (最高优先级)
2. `pretrain/dinov2_vits14_pretrain.pth`
3. `~/.cache/dinov2/dinov2_vits14_pretrain.pth`
4. `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)`

## 常见维度错误及解决方案

### 错误1: 特征数量不匹配
```
ValueError: 特征数量不匹配: ResNet输出 X 个特征图, DINOv2输出 Y 个特征图
```
**解决方案**: 检查ResNet的`out_indices`配置，确保输出4个特征图

### 错误2: Batch维度不匹配
```
ValueError: Batch维度不匹配: ResNet特征 X, DINOv2特征 Y
```
**解决方案**: 检查输入图像的batch维度是否一致

### 错误3: 通道维度不匹配
```
ValueError: 融合层通道数不匹配
```
**解决方案**: 
- 检查ResNet的`depth`配置 (应为50)
- 检查DINOv2的`embed_dim`配置 (应为384 for VIT-S)
- 检查融合层的初始化代码

## 验证代码

在模型初始化后，可以运行以下代码验证维度:

```python
# 创建测试输入
test_img = torch.randn(1, 8*6, 3, 256, 704)  # [B, N*T, C, H, W]
test_img = test_img.view(-1, 3, 256, 704)  # [BNT, C, H, W]

# 提取特征
img_feats = model.img_backbone(test_img)
semantic_feats, _ = model.dinov2_adapter(test_img)

# 验证维度
print(f"ResNet特征数量: {len(img_feats)}")
print(f"DINOv2特征数量: {len(semantic_feats)}")

for i in range(len(img_feats)):
    print(f"\n特征层级 {i}:")
    print(f"  ResNet: {img_feats[i].shape}")
    print(f"  DINOv2: {semantic_feats[i].shape}")
    print(f"  融合层输入通道: {model.semantic_fusion[i].conv.in_channels}")
    print(f"  融合层输出通道: {model.semantic_fusion[i].conv.out_channels}")
```

## 配置示例

```python
# ResNet50配置
img_backbone = dict(
    type='ResNet',
    depth=50,  # 确保是50
    num_stages=4,
    out_indices=(0, 1, 2, 3),  # 输出4个特征图
    # ...
)

# DINOv2适配器配置
dinov2_adapter = dict(
    type='DinoAdapter',
    num_heads=6,  # VIT-S
    embed_dim=384,  # VIT-S的embed_dim
    # ...
)
```

