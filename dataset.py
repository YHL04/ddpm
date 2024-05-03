

import torch
import torchvision
from torchvision import transforms


def load_dataset(img_size=64, device="cuda"):
    """
    Returns data after applying appropriate transformations,
    to work with diffusion models.
    """

    data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
        transforms.Lambda(lambda t: t.to(device))
    ])
    target_transform = transforms.Compose([
        transforms.Lambda(lambda t: torch.tensor(t).to(device))
    ])

    data = torchvision.datasets.CIFAR100(root=".", download=True,
                                         transform=data_transform,
                                         target_transform=target_transform)

    return data
