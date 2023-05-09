

from torch.optim import Adam
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import time

from diffusion import *
from model import *
from utils import *


IMG_SIZE = 64
BATCH_SIZE = 128
device = "cuda"

data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

image = next(iter(dataloader))[0]


# plots forward diffusion process
# num_images = 10
# step_size = int(T/num_images)
#
# for idx in range(0, T, step_size):
#     t = torch.Tensor([idx]).type(torch.int64)
#
#     plt.subplot(1, num_images+1, int((idx/step_size) + 1))
#     image, noise = forward_diffusion_sample(image, t)
#     show_tensor_image(image)
#
# plt.show()

"""
NOTES:

Training Step:

1.) Forward diffusion with random time-step
    (Equation: sqrt(alpha_prod) x_0  +  sqrt(1 - alpha_prod) e_0

2.) Use model to predict noise from noised image
    (Equation: pred_e = f(x|@))

Denoising:

1.) Loop for T timesteps

2.) Reverse noise predicted by model for each step
    (Equation: model_mean = sqrt(1 / alpha) * (x - beta * f(x|@) / sqrt(1 - alpha_prod))
    (Equation: model_mean + sqrt(posterior_variance) * e)

"""


def get_loss(model, x_0, t):
    batch_size = x_0.size(0)

    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)

    assert x_0.shape == (batch_size, 3, IMG_SIZE, IMG_SIZE)
    assert noise.shape == (batch_size, 3, IMG_SIZE, IMG_SIZE)
    assert noise_pred.shape == (batch_size, 3, IMG_SIZE, IMG_SIZE)

    return F.l1_loss(noise, noise_pred)


model = Unet()
model.to(device)
optimizer = Adam(model.parameters(), lr=3e-4)
epochs = 100

plt.figure(figsize=(15, 15))
plt.axis('off')

start = time.time()
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
        loss = get_loss(model, batch[0], t)
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")

        if step % 50 == 0:
            plt.clf()
            sample_plot_image(model)
            plt.pause(0.01)

