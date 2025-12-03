#!/bin/bash
# CUDA 扩展编译安装脚本
# 用于编译 RaCFormer 和 DINOv2 adapter 所需的 CUDA 扩展

echo "========================================"
echo "RaCFormer + DINOv2 CUDA 扩展编译脚本"
echo "========================================"

# 检查 CUDA 是否可用
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "❌ 错误: CUDA 不可用，请检查 PyTorch 和 CUDA 安装"
    exit 1
fi

echo "✅ CUDA 可用"
python -c "import torch; print(f'   PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'   CUDA版本: {torch.version.cuda}')"

# 1. 编译 RaCFormer 的 BEV pooling CUDA 扩展
echo ""
echo "========================================"
echo "步骤 1: 编译 RaCFormer BEV pooling CUDA 扩展"
echo "========================================"

if [ -d "models/csrc" ]; then
    cd models/csrc
    echo "📁 当前目录: $(pwd)"
    
    if python setup.py build_ext --inplace; then
        echo "✅ RaCFormer BEV pooling CUDA 扩展编译成功"
    else
        echo "❌ RaCFormer BEV pooling CUDA 扩展编译失败"
        cd ../..
        exit 1
    fi
    
    cd ../..
else
    echo "⚠️  警告: models/csrc 目录不存在，跳过"
fi

# 2. 编译 DINOv2 adapter 的 Multi-Scale Deformable Attention CUDA 扩展
echo ""
echo "========================================"
echo "步骤 2: 编译 DINOv2 Multi-Scale Deformable Attention CUDA 扩展"
echo "========================================"

if [ -d "models/backbones/nets/ops" ]; then
    cd models/backbones/nets/ops
    echo "📁 当前目录: $(pwd)"
    
    # 清理旧的编译文件
    echo "🧹 清理旧的编译文件..."
    rm -rf build dist *.egg-info
    
    # 编译并安装
    echo "🔨 开始编译..."
    if python setup.py build_ext --inplace; then
        echo "✅ 编译成功"
        
        echo "📦 安装扩展..."
        if python setup.py install; then
            echo "✅ MultiScaleDeformableAttention CUDA 扩展安装成功"
        else
            echo "❌ 安装失败"
            cd ../../../..
            exit 1
        fi
    else
        echo "❌ MultiScaleDeformableAttention CUDA 扩展编译失败"
        cd ../../../..
        exit 1
    fi
    
    cd ../../../..
else
    echo "❌ 错误: models/backbones/nets/ops 目录不存在"
    exit 1
fi

# 3. 验证安装
echo ""
echo "========================================"
echo "步骤 3: 验证 CUDA 扩展安装"
echo "========================================"

echo "🔍 验证 MultiScaleDeformableAttention..."
if python -c "import MultiScaleDeformableAttention; print('✅ MultiScaleDeformableAttention 导入成功')" 2>/dev/null; then
    echo "✅ MultiScaleDeformableAttention 可用"
else
    echo "❌ MultiScaleDeformableAttention 导入失败"
    echo ""
    echo "尝试手动安装:"
    echo "cd models/backbones/nets/ops"
    echo "python setup.py install"
    exit 1
fi

echo ""
echo "🔍 验证 DinoAdapter..."
if python -c "from models.backbones import DinoAdapter; print('✅ DinoAdapter 导入成功')" 2>&1 | grep -q "✅"; then
    echo "✅ DinoAdapter 可用"
else
    echo "⚠️  DinoAdapter 导入有警告，但可能仍然可用"
fi

echo ""
echo "========================================"
echo "✅ CUDA 扩展编译安装完成！"
echo "========================================"
echo ""
echo "下一步: 运行完整检查"
echo "python check_dinov2_setup.py"

