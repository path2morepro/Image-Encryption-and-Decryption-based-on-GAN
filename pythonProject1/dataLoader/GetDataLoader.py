from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from sklearn.model_selection import train_test_split
from dataLoader.MyDataset import Dataset


def GetDataLoader(data_path, image_size, test_path=None, batch_size=16):
    if test_path is not None:
        data_path = test_path

    sharp_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,))         # 适用于灰度图像
    ])

    full_dataset = Dataset(data_path, img_size=image_size, sharp_transform=sharp_transform)

    training_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        pin_memory=True
    )

    return training_loader
