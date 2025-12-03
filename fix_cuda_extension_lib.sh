#!/bin/bash
# 修复 CUDA 扩展库路径问题

echo "=========================================="
echo "  修复 CUDA 扩展库路径"
echo "=========================================="

# 添加 PyTorch 库路径到 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")

echo "已设置库路径: $LD_LIBRARY_PATH"
echo ""

echo "测试 CUDA 扩展导入..."
python -c "import MultiScaleDeformableAttention; print('✅ CUDA扩展加载成功')"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 修复成功！"
    echo "=========================================="
    echo ""
    echo "请将以下命令添加到您的 ~/.bashrc 或 ~/.bash_profile 中："
    echo ""
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$(python -c \"import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))\")"
    echo ""
    echo "或者在每次运行前执行："
    echo "source fix_cuda_extension_lib.sh"
else
    echo ""
    echo "❌ 仍然无法加载 CUDA 扩展"
    echo "请检查 PyTorch 安装是否正确"
fi


