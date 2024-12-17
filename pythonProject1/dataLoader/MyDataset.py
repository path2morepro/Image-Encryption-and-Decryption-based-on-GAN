import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class Dataset(Dataset):
    def __init__(self, data_dir, img_size, sharp_transform=None):
        # self.sharp = os.path.join(data_dir, "sharp")
        self.data_dir = data_dir
        self.img_size = img_size
        self.sharp_transform = sharp_transform
        self.key = None
        if len(os.listdir(self.data_dir)) > 50000:
            self.sharp_image_list = os.listdir(self.data_dir)[:95000]
        else:
            self.sharp_image_list = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.sharp_image_list)

    def __getitem__(self, idx):
        # 加载清晰图片
        global sharp_transformed
        sharp_image_path = os.path.join(self.data_dir, self.sharp_image_list[idx])
        sharp_image = Image.open(sharp_image_path).convert('L')

        # # 为每张图片生成一个随机图像当作密钥
        key = torch.randint(0, 256, (1, self.img_size, self.img_size), dtype=torch.float) / 255.0  # 灰度图像密钥

        # 对清晰图片应用变换
        if self.sharp_transform is not None:
            sharp_transformed = self.sharp_transform(sharp_image)

        # Transform the key to match the image size if needed
        # Here, assuming the key is already in the correct size

        return sharp_transformed, key


