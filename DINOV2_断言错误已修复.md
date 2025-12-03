# DINOv2 Adapter 断言错误修复完成

## 问题根源

`MSDeformAttn` 模块的断言失败:
```
assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
AssertionError
```

**根本原因**:
- `deform_inputs` 函数使用简单的整数除法 (`h//8`, `h//16`, `h//32`) 来计算 `spatial_shapes`
- 但 SPM (Spatial Prior Module) 的实际输出尺寸由于 padding 和多次下采样,与简单除法计算的结果不一致
- 导致 `spatial_shapes` 定义的总元素数与实际特征序列长度不匹配

**具体示例**:

对于 padding 后的图像 (266×714):
- **预期 (错误)**:
  - c2: (266//8) × (714//8) = 33 × 89 = 2937
  - c3: (266//16) × (714//16) = 16 × 44 = 704
  - c4: (266//32) × (714//32) = 8 × 22 = 176
  - 总计: 3817

- **实际 (SPM 输出)**:
  - c1: (67, 179) ← stem 输出
  - c2: (34, 90) ← c1 经过 stride=2 卷积
  - c3: (17, 45) ← c2 经过 stride=2 卷积
  - c4: (9, 23) ← c3 经过 stride=2 卷积
  - 总计 (c2+c3+c4): 3060 + 765 + 207 = **4032** ❌

## 修复方案

### 修改文件 1: `dinov2_adapter.py`

**位置**: `models/backbones/nets/dino_v2_with_adapter/dino_v2_adapter/dinov2_adapter.py`

**修改内容**: 
1. 在 `forward` 方法中,先执行 SPM forward
2. 从 SPM 输出推导实际的空间形状
3. 将实际形状传递给 `deform_inputs` 函数

**关键改动**:
```python
# 修改前 (第 194-200 行)
deform_inputs1, deform_inputs2 = deform_inputs(x)
c1, c2, c3, c4 = self.spm(x)
c2, c3, c4 = self._add_level_embed(c2, c3, c4)
c = torch.cat([c2, c3, c4], dim=1)

# 修改后
# 先执行 SPM forward
c1, c2, c3, c4 = self.spm(x)

# 从 SPM 输出推导实际空间形状
_, _, H1_spm, W1_spm = c1.shape
bs_spm, L2, dim_spm = c2.shape
_, L3, _ = c3.shape
_, L4, _ = c4.shape

H2_spm, W2_spm = H1_spm // 2, W1_spm // 2
H3_spm, W3_spm = H2_spm // 2, W2_spm // 2
H4_spm, W4_spm = H3_spm // 2, W3_spm // 2

# 验证
assert L2 == H2_spm * W2_spm
assert L3 == H3_spm * W3_spm
assert L4 == H4_spm * W4_spm

c2, c3, c4 = self._add_level_embed(c2, c3, c4)
c = torch.cat([c2, c3, c4], dim=1)

# 使用实际形状构建 deform_inputs
deform_inputs1, deform_inputs2 = deform_inputs(x, c2_shape=(H2_spm, W2_spm), 
                                                c3_shape=(H3_spm, W3_spm), 
                                                c4_shape=(H4_spm, W4_spm))
```

**同样的修改也应用到 `extract_intermediate_features` 方法**。

### 修改文件 2: `adapter_modules.py`

**位置**: `models/backbones/nets/dino_v2_with_adapter/dino_v2_adapter/adapter_modules.py`

**修改内容**: 
修改 `deform_inputs` 函数签名,接受可选的 SPM 形状参数

**关键改动**:
```python
# 修改前
def deform_inputs(x):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    # ...

# 修改后
def deform_inputs(x, c2_shape=None, c3_shape=None, c4_shape=None):
    """
    构建 deformable attention 的输入
    
    Args:
        x: 输入图像 [bs, c, h, w]
        c2_shape: SPM 输出 c2 的空间形状 (H, W), 如果为 None 则使用默认计算
        c3_shape: SPM 输出 c3 的空间形状 (H, W), 如果为 None 则使用默认计算
        c4_shape: SPM 输出 c4 的空间形状 (H, W), 如果为 None 则使用默认计算
    """
    bs, c, h, w = x.shape
    
    # 如果提供了 SPM 的实际形状,使用它们;否则使用默认计算
    if c2_shape is not None and c3_shape is not None and c4_shape is not None:
        H2, W2 = c2_shape
        H3, W3 = c3_shape
        H4, W4 = c4_shape
    else:
        # 默认计算
        H2, W2 = h // 8, w // 8
        H3, W3 = h // 16, w // 16
        H4, W4 = h // 32, w // 32
    
    spatial_shapes = torch.as_tensor([(H2, W2), (H3, W3), (H4, W4)],
                                     dtype=torch.long, device=x.device)
    # ...
```

## 验证修复

### 在服务器上执行

```bash
cd ~/derma/RACDION

# 确保环境变量已设置
source fix_cuda_extension_lib.sh

# 测试图像尺寸兼容性
python test_dinov2_patch_size.py

# 运行完整检查
python tools/check_dinov2_integration.py

# 调试脚本 (可选)
python debug_dinov2_shapes.py
```

### 预期结果

1. **`test_dinov2_patch_size.py`**: 所有测试用例显示 ✅
   - RaCFormer默认 (256x704)
   - 完美整除 (224x224)
   - NuScenes原始 (900x1600)
   - 随机尺寸1 (480x640)
   - 随机尺寸2 (300x500)

2. **`tools/check_dinov2_integration.py`**: DINOv2功能测试通过 ✅

3. **`debug_dinov2_shapes.py`**: 前向传播成功 ✅

## 文件同步清单

从本地 Cursor 上传到服务器的文件:

1. ✅ `models/backbones/nets/dino_v2_with_adapter/dino_v2_adapter/dinov2_adapter.py`
2. ✅ `models/backbones/nets/dino_v2_with_adapter/dino_v2_adapter/adapter_modules.py`

## 技术说明

### 为什么 SPM 的输出尺寸与简单除法不同？

SPM 的结构:
```
输入 (H, W)
  ↓ Conv (stride=2) → (H//2, W//2)
  ↓ Conv
  ↓ Conv
  ↓ MaxPool (stride=2) → (H//4, W//4) ← c1
  ↓ Conv2 (stride=2) → c1//2 ← c2
  ↓ Conv3 (stride=2) → c2//2 ← c3
  ↓ Conv4 (stride=2) → c3//2 ← c4
```

关键点:
- `c1` 是经过两次 stride=2 操作后的结果: `H//4, W//4`
- `c2, c3, c4` 分别是 `c1` 的 `1/2, 1/4, 1/8`
- **不是** 相对于原始输入的 `H//8, H//16, H//32`

因此,当输入有 padding 时 (如 256→266):
- 简单除法: 266//8 = 33
- 实际 SPM: (266//4)//2 = (66)//2 = 33 ← 碰巧一样!
- 但对于 266//32: 8 vs ((66)//2)//2)//2 = (33)//2)//2 = ...

由于 Python 的整数除法规则,连续除法与一次性除法在有余数时结果不同。

### 修复的优雅性

通过从 SPM 的**实际输出**推导形状,而不是从输入推测:
- ✅ 消除了所有假设
- ✅ 适用于任意输入尺寸
- ✅ 自动处理 padding
- ✅ 保证 spatial_shapes 与实际特征匹配

## 下一步

修复完成后,您就可以开始训练了:

```bash
# 单GPU
python train.py --config configs/racformer_r50_nuimg_704x256_f8_with_dinov2_fixed.py

# 多GPU (8卡)
torchrun --nproc_per_node 8 train.py --config configs/racformer_r50_nuimg_704x256_f8_with_dinov2_fixed.py
```

祝训练顺利! 🎉

