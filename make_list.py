import os
import glob

# ================= 配置区域 =================
# 这里直接使用你提供的绝对路径
DATA_ROOT = '/home/ub/pql/Swin-Unet-main/data/TN3K'

# 输出列表文件的路径 (会自动创建文件夹)
OUTPUT_DIR = '/home/ub/pql/Swin-Unet-main/lists/TN3K'


# ===========================================

def generate_list(phase):
    """
    phase: 'train' 或 'test'
    """
    # 关键点：脚本会去找 DATA_ROOT/train/images 这个文件夹
    img_dir = os.path.join(DATA_ROOT, phase, 'images')

    # 检查文件夹是否存在
    if not os.path.exists(img_dir):
        print(f"Error: 找不到图片目录！脚本正在寻找: {img_dir}")
        print(f"请检查你的 {phase} 文件夹下是否有 'images' 子文件夹！")
        return

    # 获取所有图片文件
    files = sorted(os.listdir(img_dir))

    # 支持常见后缀
    img_files = [f for f in files if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]

    if len(img_files) == 0:
        print(f"Warning: 在 {img_dir} 下没有找到图片文件！")
        return

    # 准备输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    out_path = os.path.join(OUTPUT_DIR, f'{phase}.txt')

    print(f"正在处理 {phase} 集...")
    print(f"  - 源目录: {img_dir}")
    print(f"  - 找到图片: {len(img_files)} 张")

    with open(out_path, 'w') as f:
        for filename in img_files:
            # 去除后缀 (例如: "image_01.jpg" -> "image_01")
            name_no_ext = os.path.splitext(filename)[0]
            f.write(name_no_ext + '\n')

    print(f"  - 列表已生成: {out_path}\n")


if __name__ == '__main__':
    generate_list('train')
    generate_list('test')
    print("全部完成！请检查 lists/TN3K 文件夹。")