from dataloaders import CustomDataLoader
import torch
from models import VariationalAutoEncoder
import yaml

with open("configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)

print(config)
x = torch.randn((3, 3, 250, 250))
print(x.shape)
vae = VariationalAutoEncoder(config)
# print(vae(x)[0].shape)
# print(vae)
dataloader = CustomDataLoader(400)
print(len(dataloader.train_dataset))
print(len(dataloader.val_dataset))
