from dataloaders import CustomDataLoader
import torch
from models import VariationalAutoEncoder
import wandb
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
# wandb.login()

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # Name of the metric to monitor
    dirpath='checkpoints/',  # Directory where the checkpoints will be saved
    filename='best-checkpoint',  # Checkpoint file name
    save_top_k=1,  # Save the top k models
    mode='min',  # 'min' mode saves the model when the monitored quantity stops decreasing
    verbose=True
)
wandb_logger = WandbLogger(project="Art_Proj")

with open("configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)

model = VariationalAutoEncoder(config)
dataloader = CustomDataLoader(batch_size=10)

trainer = Trainer(max_epochs=10,
                  callbacks=checkpoint_callback,
                  logger=wandb_logger)
# trainer.tune.lr_find(model)
trainer.fit(model, dataloader)
# print(config)
# x = torch.randn((3, 3, 250, 250))
# print(x.shape)

# # print(vae(x)[0].shape)
# # print(vae)

# print(len(dataloader.train_dataset))
# print(len(dataloader.val_dataset))
