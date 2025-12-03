#!/bin/bash
# DINOv2 导入问题一键修复脚本

set -e

echo "=========================================="
echo "  DINOv2 导入问题修复脚本"
echo "=========================================="

# 确保在项目根目录
if [ ! -f "train.py" ]; then
    echo "❌ 错误: 请在项目根目录运行此脚本"
    exit 1
fi

echo ""
echo "当前目录: $(pwd)"
echo "Python: $(which python)"
echo "CUDA: $(nvcc --version | grep release | awk '{print $5}' | cut -d',' -f1)"
echo ""

echo "步骤 1/4: 编译 MultiScaleDeformableAttention CUDA 扩展..."
echo "----------------------------------------"
cd models/backbones/nets/ops
python setup.py build install
cd ../../../../

echo ""
echo "步骤 2/4: 验证 CUDA 扩展安装..."
echo "----------------------------------------"
python -c "import MultiScaleDeformableAttention; print('✅ CUDA 扩展安装成功')" || {
    echo "❌ CUDA 扩展安装失败，请检查编译错误"
    exit 1
}

echo ""
echo "步骤 3/4: 测试 DinoAdapter 导入..."
echo "----------------------------------------"
python -c "from models.backbones import DinoAdapter; print('✅ DinoAdapter 可导入')" || {
    echo "❌ DinoAdapter 导入失败"
    exit 1
}

echo ""
echo "步骤 4/4: 运行完整检查..."
echo "----------------------------------------"
if [ -f "check_dinov2_setup.py" ]; then
    python check_dinov2_setup.py
else
    echo "⚠️  check_dinov2_setup.py 不存在，跳过完整检查"
fi

echo ""
echo "=========================================="
echo "  ✅ 所有问题已修复！"
echo "=========================================="
echo ""
echo "现在可以开始训练:"
echo "  torchrun --nproc_per_node 8 train.py --config configs/racformer_r50_nuimg_704x256_f8_with_dinov2_fixed.py"
echo ""
echo "或使用分布式训练脚本:"
echo "  bash dist_train.sh configs/racformer_r50_nuimg_704x256_f8_with_dinov2_fixed.py 8"
echo ""

