# DINOv2 Adapter 断言错误最终修复方案

## 问题根源

在 `models/backbones/nets/dino_v2_with_adapter/dino_v2_adapter/dinov2_adapter.py` 的 `forward` 方法中：

```python
# SPM forward
c1, c2, c3, c4 = self.spm(x)
c2, c3, c4 = self._add_level_embed(c2, c3, c4)
c = torch.cat([c2, c3, c4], dim=1)  # 拼接特征
```

SPM 的输出：
- `c1`: `[bs, dim, H//4, W//4]` (4D, **未** flatten)
- `c2`: `[bs, L2, dim]` (3D, 已 flatten,  L2 = (H//8)×(W//8))
- `c3`: `[bs, L3, dim]` (3D, 已 flatten, L3 = (H//16)×(W//16))
- `c4`: `[bs, L4, dim]` (3D, 已 flatten, L4 = (H//32)×(W//32))

拼接后的 `c`: `[bs, L2+L3+L4, dim]`

但在 `deform_inputs` 函数中，`deform_inputs1` 的 `spatial_shapes` 定义为：
```python
spatial_shapes = torch.as_tensor([
    (h // 8, w // 8),
    (h // 16, w // 16),
    (h // 32, w // 32)
], dtype=torch.long, device=x.device)
```

这表示 `feat` 应该包含 **3个未 flatten 的特征图**，但实际上 `c` 是一个 **单一的 flattened 序列**。

## 修复方案

修改 `dinov2_adapter.py` 中的 `forward` 方法，不要提前拼接 `c2, c3, c4`，而是让它们保持独立，然后在 `InteractionBlock` 中根据 `spatial_shapes` 动态拼接。

但更简单的方法是：**修改 `Injector` 的调用方式，使用正确的 `spatial_shapes`**。

### 方案 A: 修改 `dinov2_adapter.py` (推荐)

在 `forward` 方法中，不要拼接 `c`，而是保持 tuple 形式传递给 `InteractionBlock`：

**修改文件**: `models/backbones/nets/dino_v2_with_adapter/dino_v2_adapter/dinov2_adapter.py`

**修改位置**: 第 197-200 行

**修改前**:
```python
# SPM forward
c1, c2, c3, c4 = self.spm(x)
c2, c3, c4 = self._add_level_embed(c2, c3, c4)
c = torch.cat([c2, c3, c4], dim=1)
```

**修改后**:
```python
# SPM forward
c1, c2, c3, c4 = self.spm(x)
c2, c3, c4 = self._add_level_embed(c2, c3, c4)
# 保持为独立特征，供 InteractionBlock 使用
c = torch.cat([c2, c3, c4], dim=1)  # 拼接成单一序列
```

**等等，这不对！** 问题是 `c` 已经是拼接后的序列，但 `spatial_shapes` 定义的是 3 个独立特征图。

让我重新检查 `Injector` 的实现...

实际上，`MSDeformAttn` 期望的输入是：
- `value` (即 `feat`): `[bs, sum(H_i × W_i), dim]` - 所有层级特征 flatten 后拼接
- `spatial_shapes`: `[num_levels, 2]` - 每个层级的 (H, W)

所以目前的实现应该是正确的！问题可能在于 **SPM 输出的特征序列长度计算错误**。

让我重新分析...

### 真正的问题

从调试输出：
```
deform_inputs1 spatial_shapes (SPM 特征):
  Level 0: 33x89
  Level 1: 16x44
  Level 2: 8x22
  Total elements: 3817
```

计算：33×89 + 16×44 + 8×22 = 2937 + 704 + 176 = **3817**

但 `c` 的实际序列长度是多少呢？让我们计算：

对于 padding 后的图像 (266×714):
- `c2`: (266//8) × (714//8) = 33 × 89 = 2937
- `c3`: (266//16) × (714//16) = 16 × 44 = 704  
- `c4`: (266//32) × (714//32) = 8 × 22 = 176

总计: 2937 + 704 + 176 = **3817**

理论上应该是匹配的！

但等等... SPM 的 forward 中，`c2, c3, c4` 的计算涉及多次 stride=2 的卷积。让我重新检查 SPM 的输出尺寸计算...

### 最终真相

SPM 的结构：
1. `stem`: 3→2 (stride=2 MaxPool)  → 输出: H//4, W//4
2. `conv2`: stem → c2 (stride=2) → 输出: H//8, W//8
3. `conv3`: c2 → c3 (stride=2) → 输出: H//16, W//16
4. `conv4`: c3 → c4 (stride=2) → 输出: H//32, W//32

但是！`stem` 中有一个 MaxPool with stride=2，而之前还有 stride=2 的卷积...

让我精确计算 SPM 的输出尺寸：

对于输入 (266, 714):
1. Conv stride=2: → (133, 357)
2. MaxPool stride=2: → (67, 179) ← 这是 c1
3. conv2 stride=2: → (34, 90) ← 这是 c2  
4. conv3 stride=2: → (17, 45) ← 这是 c3
5. conv4 stride=2: → (9, 23) ← 这是 c4

**这才是真正的输出尺寸！**

但 `deform_inputs` 使用的是：
- (266//8, 714//8) = (33, 89)
- (266//16, 714//16) = (16, 44)
- (266//32, 714//32) = (8, 22)

**这就是不匹配的原因！**

## 最终修复方案

修改 `deform_inputs` 函数，使其根据 SPM 的实际下采样率（不是简单的 //8, //16, //32）来计算 `spatial_shapes`。

或者，更简单的方法：**传递 SPM 输出的实际形状给 `deform_inputs`**。

### 实施方案

修改 `dinov2_adapter.py`，在调用 `deform_inputs` 时传递 SPM 输出的形状：

**文件**: `models/backbones/nets/dino_v2_with_adapter/dino_v2_adapter/dinov2_adapter.py`

**修改位置 1**: 第 195-200 行

**修改前**:
```python
# 2. 原始流程（使用 padding 后的 x）
deform_inputs1, deform_inputs2 = deform_inputs(x)

# SPM forward
c1, c2, c3, c4 = self.spm(x)
c2, c3, c4 = self._add_level_embed(c2, c3, c4)
c = torch.cat([c2, c3, c4], dim=1)
```

**修改后**:
```python
# 2. SPM forward (必须先执行，以获取实际特征形状)
c1, c2, c3, c4 = self.spm(x)
c2, c3, c4 = self._add_level_embed(c2, c3, c4)
c = torch.cat([c2, c3, c4], dim=1)

# 3. 使用 SPM 输出的实际形状计算 deform_inputs
# c2: [bs, L2, dim], c3: [bs, L3, dim], c4: [bs, L4, dim]
bs_spm, L2, dim_spm = c2.shape
_, L3, _ = c3.shape
_, L4, _ = c4.shape

# 反推空间形状 (假设是方形或接近方形)
import math
H2, W2 = c2.shape[2], c2.shape[3] if len(c2.shape) == 4 else (int(math.sqrt(L2)), L2 // int(math.sqrt(L2)))
```

等等，这又不对了。`c2, c3, c4` 已经是 3D 的了 (flattened)，我们无法直接获取 H, W。

让我重新看 SPM 的代码...

