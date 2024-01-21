# LEARNING_RATE = 4e-3  # karpathy constant
# DROPOUT = 0.4
import yaml

with open("configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)
print(config["model_config"]["encoder_layers"])
print(config["train_config"])
# print(type(config))
