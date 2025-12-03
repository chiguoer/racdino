import os
import sys

def fix_symlink():
    # 定义目标路径和源路径
    target_data_path = '/data/dataset/RacFormer/nuscenes'
    link_dir = 'data'
    link_name = 'nuscenes'
    link_path = os.path.join(link_dir, link_name)

    # 1. 检查真实数据路径是否存在
    if not os.path.exists(target_data_path):
        print(f"❌ 错误: 真实数据路径不存在: {target_data_path}")
        print("   请确认数据究竟存放在哪里。")
        return False

    # 2. 确保 data 目录存在
    if not os.path.exists(link_dir):
        print(f"创建目录: {link_dir}")
        os.makedirs(link_dir)

    # 3. 检查软链接是否已存在
    if os.path.exists(link_path):
        if os.path.islink(link_path):
            current_target = os.readlink(link_path)
            if current_target == target_data_path:
                print(f"✅ 软链接已正确存在: {link_path} -> {target_data_path}")
                return True
            else:
                print(f"⚠️  软链接存在但指向不同: {link_path} -> {current_target}")
                print(f"   将移除并重新创建指向: {target_data_path}")
                os.unlink(link_path)
        else:
            print(f"⚠️  路径存在但不是软链接: {link_path}")
            print("   请手动检查并备份/删除该目录，以便创建软链接。")
            return False

    # 4. 创建软链接
    try:
        os.symlink(target_data_path, link_path)
        print(f"✅ 成功创建软链接: {link_path} -> {target_data_path}")
        return True
    except OSError as e:
        print(f"❌ 创建软链接失败: {e}")
        return False

if __name__ == "__main__":
    if fix_symlink():
        print("\n🎉 修复完成！现在您可以尝试重新运行训练脚本了。")
    else:
        print("\n🚫 修复未能完成，请检查上述错误。")

