from dataloaders import train_dataset
import torch
from PIL import Image
from torchvision.io import read_image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

image_path = "data/musemart/dataset_updated/training_set/painting/0012.jpg"
img = read_image(path=image_path)
print(img.shape)
print(img.min(), "min")

target_shape = [250, 250]
# a = torch.rand(3, 306, 290)
# b = torch.rand(3, 3, 3)

# print(a)
c = transforms.ToPILImage()(img)
print(c)

d = transforms.functional.resize(c, size=target_shape)
e = transforms.ToTensor()(d)
print(d)
print(e.shape)
plt.imshow(img.permute(1, 2, 0))
# plt.show()
# plt.imshow(e.permute(1, 2, 0))
# print(train_dataset)
# plt.show()
