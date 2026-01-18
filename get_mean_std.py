import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# === 1. 配置路径 (请重点检查 base_dir) ===
# 列表所在的文件夹路径
LIST_DIR = '/home/ub/pql/Swin-Unet-main/lists/TN3K'

# 图片所在的根目录
# 警告：请根据你的实际情况修改这里！
# 如果你的图片在 /home/ub/pql/Swin-Unet-main/data/TN3K/train/images/...
# 那么 BASE_DIR 应该是 /home/ub/pql/Swin-Unet-main/data/TN3K
BASE_DIR = '/home/ub/pql/Swin-Unet-main/data/TN3K'

# 目标尺寸 (Swin-Unet)
IMG_SIZE = (224, 224)


# === 2. 辅助函数：Letterbox Resize (保持比例填充) ===
def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))  # 填充灰色
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


# === 3. 数据集定义 ===
class SimpleTN3K(Dataset):
    def __init__(self, base_dir, list_dir, split="train"):
        self.base_dir = base_dir
        self.split = split

        # 读取 train.txt
        list_path = os.path.join(list_dir, self.split + '.txt')
        print(f"读取列表文件: {list_path}")

        if not os.path.exists(list_path):
            raise FileNotFoundError(f"找不到列表文件: {list_path}")

        with open(list_path, 'r') as f:
            self.sample_list = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx]

        # 拼接图片路径： base_dir/train/images/xxx.jpg
        # 请确认你的图片后缀是 .jpg 还是 .png
        img_path = os.path.join(self.base_dir, self.split, "images", slice_name + ".jpg")

        if not os.path.exists(img_path):
            # 容错：如果找不到 jpg，尝试找 png
            img_path = img_path.replace(".jpg", ".png")
            if not os.path.exists(img_path):
                print(f"警告: 找不到文件 {img_path}")
                # 返回全黑图片防止崩溃，但在计算均值时会有微小误差
                return np.zeros((224, 224, 3), dtype=np.float32)

        img = Image.open(img_path).convert('RGB')

        # 使用 Letterbox 保持一致性
        img = letterbox_image(img, IMG_SIZE)

        # 归一化到 0-1
        img = np.array(img).astype(np.float32) / 255.0
        return img


# === 4. 主计算逻辑 ===
def compute():
    if not os.path.exists(BASE_DIR):
        print(f"错误: 找不到图片数据目录 BASE_DIR: {BASE_DIR}")
        print("请打开脚本修改 BASE_DIR 变量。")
        return

    dataset = SimpleTN3K(BASE_DIR, LIST_DIR, split="train")

    # num_workers 可以根据你的 CPU 核心数调整，太高可能会卡
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    mean = 0.
    std = 0.
    nb_samples = 0.

    print(f"开始计算... 数据集大小: {len(dataset)}")

    for data in tqdm(loader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(3), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print("\n" + "=" * 40)
    print("计算完成！请将以下代码替换到 dataset_tn3k.py 的 __init__ 中:")
    print("=" * 40)
    print(f"self.mean = [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"self.std  = [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    print("=" * 40)


if __name__ == '__main__':
    compute()