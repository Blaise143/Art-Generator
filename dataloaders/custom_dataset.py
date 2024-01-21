import torch
from torchvision import transforms, datasets
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class CustomDataset(LightningDataModule):
    def __init__(self, batch_size: int,
                 train_path: str = "data/musemart/dataset_updated/training_set",
                 val_path: str = "data/musemart/dataset_updated/validation_set") -> None:
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size

    def train_dataloader(self):

        train_dataset = datasets.ImageFolder(
            root=self.train_path,
            transform=self.transform
        )
        dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def val_dataloader(self):
        val_dataset = datasets.ImageFolder(
            root=self.val_path,
            transform=self.transform
        )
        dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        return dataloader
