

import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


def show_tensor_image(image):
    """
    :param image: Tensor[batch_size, channels, height, width]

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
