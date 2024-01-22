import torchvision.transforms as transforms
import torch
from typing import Union
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.io import read_image


def plot_image(img: Union[torch.Tensor, str]):
    assert type(img) in [torch.Tensor, str], "Img should be a path or a tensor"
    if isinstance(img, str):
        img = read_image(path=img)

    target_shape = [250, 250]
    pil_img = transforms.ToPILImage()(img)
    resized_img = transforms.functional.resize(pil_img, target_shape)
    tensored_img = transforms.ToTensor()(resized_img)
    plt.imshow(tensored_img.permute(1, 2, 0))
    plt.show()
    return plt


image_path = "data/musemart/dataset_updated/training_set/painting/0012.jpg"
img_read = read_image(image_path)
plot_image(image_path)
