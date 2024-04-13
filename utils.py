

import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


def show_tensor_image(image, save=None):
    """
    :param image: Tensor[batch_size, channels, height, width]
    :param save: path to saved file (default = None)

    Plots image after applying reverse transformations.

    CHW to HWC
    """
    B, C, H, W = image.shape

    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    if len(image.shape) == 4:
        image = image[0, :, :, :]

    image = reverse_transforms(image)
    image = image.resize((H*2, W*2))

    plt.imshow(image)

    if save is not None:
        image.save(save)

