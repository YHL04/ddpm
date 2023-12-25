

import torch
import torch.nn.functional as F
from torch.optim import Adam

import math
import matplotlib.pyplot as plt

from unet import Unet
from utils import show_tensor_image


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """
    linear schedule, proposed in original ddpm paper
    """
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps, s=0.008, beta_max=0.999):
    """
    :param s: 0.0008 (default)

    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    also refer to https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf
    """
    cosine_schedule = lambda t: torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    t = torch.linspace(0, timesteps, timesteps+1, dtype=torch.float32) / timesteps
    t1, t2 = t[:-1], t[1:]

    betas = 1 - cosine_schedule(t2) / cosine_schedule(t1)
    betas = torch.clip(betas, 0, beta_max)

    return betas


class DDPM:

    def __init__(self, T=1000, img_size=64, agg_grad=4, lr=1e-4, device="cuda"):

        # general parameters
        self.T = T
        self.img_size = img_size
        self.device = device

        self.agg_grad = agg_grad
        self.t = 0

        # model
        self.model = Unet().to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)

        # plot parameters
        plt.figure(figsize=(15, 15))
        plt.axis("off")

        # precomputed variables
        self.betas = cosine_beta_schedule(timesteps=T)
        self.alphas = 1. - self.betas

        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)


    @staticmethod
    def get_index_from_list(vals, t, x_shape):
        """
        :param vals   : Tensor[500]
        :param t      : Tensor[1]
        :param x_shape: torch.Size()

        :return: output: Tensor[B, 1, 1, 1]

        Obtain value associated with timestep for different precomputed variables
        and return repeat of value with shape [B, 1, 1, 1]
        """
        B = t.shape[0]
        output = vals.gather(-1, t.cpu())
        output = output.reshape(B, *((1,) * (len(x_shape) - 1))).to(t.device)

        return output

    def get_loss(self, x_0, t):
        """
        Perform forward step to get x_t and
        pass through model to approximate x_noisy
        """
        x_noisy, noise = self.forward_step(x_0, t)
        noise_pred = self.model(x_noisy, t)

        return F.l1_loss(noise, noise_pred)

    def train_step(self, batch):
        B = batch.size(0)

        t = torch.randint(0, self.T, (B,), device=self.device).long()
        loss = self.get_loss(batch, t) / self.agg_grad
        loss.backward()

        if self.t % self.agg_grad == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.t += 1

        return loss.item()

    @torch.no_grad()
    def forward_step(self, x_0, t):
        """
        TODO: Math needs verification

        forward step to any t from x_0 given by the equation

        x_t = sqrt(alpha_prod) * x_0 + sqrt(1 - alpha_prod) * e_0
        """
        x_0 = x_0.to(self.device)
        noise = torch.randn_like(x_0).to(self.device)

        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.to(self.device)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.to(self.device)

        mean = sqrt_alphas_cumprod_t * x_0
        variance = sqrt_one_minus_alphas_cumprod_t * noise

        return mean + variance, noise

    @torch.no_grad()
    def backward_step(self, x, t):
        """
        TODO: Needs verification

        One denoising step using model mean prediction and precomputed variance
        e = noise with N(0, 1)

        model_mean = sqrt(1/alphas) * (x - betas * pred_e / sqrt(1 - alpha_prod))
        variance = sqrt(posterior_variance) * e

        reparameterization trick?
        x_t-1 = model_mean + variance
        """
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x.shape)

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t == 0: return model_mean

        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)

        return model_mean + torch.sqrt(posterior_variance_t) * noise

    def plot_denoising_process(self, num_images=10, save=None):
        """
        t = 0 is image
        t = 300 is noise
        """
        img = torch.randn((1, 3, self.img_size, self.img_size), device=self.device)
        step_size = int(self.T/num_images)

        for i in range(0, self.T)[::-1]:
            t = torch.full((1,), i, device=self.device, dtype=torch.long)
            img = self.backward_step(img, t)

            if i % step_size == 0:
                plt.subplot(1, num_images, int(i/step_size + 1))
                show_tensor_image(img.detach().cpu(), save=save)

        plt.pause(0.01)

