import torch
from torchvision import transforms, datasets
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from PIL import UnidentifiedImageError
# from torchvision.datasets import ImageFolder


class CustomDataLoader(LightningDataModule):
    def __init__(self,
                 batch_size: int,
                 train_path: str = "data/musemart/dataset_updated/training_set",
                 val_path: str = "data/musemart/dataset_updated/validation_set") -> None:
        super().__init__()
        transform = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.train_dataset = RobustImageFolder(
            root=self.train_path,
            transform=transform
        )
        self.val_dataset = RobustImageFolder(
            root=self.val_path,
            transform=transform
        )

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
            # num_workers=7

        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=custom_collate_fn,
            shuffle=False,
            # num_workers=7
        )
        return dataloader


class RobustImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (IOError, OSError, UnidentifiedImageError) as e:
            # print(
            # f"Error encountered with image at index: {index}. Error: {e}. Skipping.")
            return None


def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))  # Remove None items
    if len(batch) == 0:  # If all items are bad in a batch
        return torch.tensor([]), torch.tensor([])
    # Use the default collate function to combine the filtered items
    return torch.utils.data.dataloader.default_collate(batch)
