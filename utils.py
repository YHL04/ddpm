
import torch
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

from diffusion import sample_timestep


T = 300
IMG_SIZE = 64
BATCH_SIZE = 128


def load_transformed_dataset():
    """
    Returns data after applying appropriate transformations,
    to work with diffusion models.
    """
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.CIFAR100(root=".", download=True,
                                          transform=data_transform)

    test = torchvision.datasets.CIFAR100(root=".", download=True,
                                         transform=data_transform)
    return torch.utils.data.ConcatDataset([train, test])


def show_tensor_image(image):
    """
    Plots image after applying reverse transformations.

    CHW to HWC
    """

    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))


@torch.no_grad()
def sample_plot_image(model, device="cuda"):
    """
    Plot out the whole de-noising process
    """
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)

    num_images = 10
    stepsize = int(T / num_images)

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(model, img, t)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i / stepsize + 1))
            show_tensor_image(img.detach().cpu())

