

import torch
import torch.nn.functional as F
from torch.optim import Adam

import math
import matplotlib.pyplot as plt

from utils import show_tensor_image


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02, device="cuda"):
    """
    linear schedule, proposed in original ddpm paper
    """
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=device)


def cosine_beta_schedule(timesteps, s=0.008, beta_max=0.999, device="cuda"):
    """
    :param s: 0.0008 (default)

    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    cosine_schedule = lambda t: torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2

    t = torch.linspace(0, timesteps, timesteps+1, dtype=torch.float32, device=device) / timesteps
    t1, t2 = t[:-1], t[1:]

    betas = 1 - cosine_schedule(t2) / cosine_schedule(t1)
    betas = torch.clip(betas, 0, beta_max)

    return betas


class DDPM:
    """
    Improved DDPM improvements according to paper:
    https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf

    Improving the log likelihood

    1. Learning
    2. Cosine Schedule
    3. Reducing Gradient Noise
    4. Improving Sampling Speed

    """

    def __init__(self, model, T, lr, img_size=64, agg_grad=1, lambda_=0.001, device="cuda"):

        # general parameters
        self.T = T
        self.img_size = img_size
        self.device = device

        self.agg_grad = agg_grad
        self.lambda_ = lambda_
        self.t = 0

        # model
        self.model = model
        self.optimizer = Adam(self.model.parameters(), lr=lr)

        # plot parameters
        plt.figure(figsize=(15, 15))
        plt.axis("off")

        # precomputed variables
        self.betas = linear_beta_schedule(timesteps=T, device=device)
        # self.betas = cosine_beta_schedule(timesteps=T, device=device)
        self.alphas = 1. - self.betas

        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.concatenate([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )

        # improved ddpm variational loss
        self.mu_term1 = torch.sqrt(self.alphas_cumprod_prev) * self.betas / (1 - self.alphas_cumprod)
        self.mu_term2 = torch.sqrt(self.alphas) * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

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
        output = vals.gather(-1, t)
        output = output.reshape(B, *((1,) * (len(x_shape) - 1))).to(t.device)

        return output

    def _get_pred_mean_var(self, x_t, e, v, t):
        """e is optimized using simple loss, so detach e from variational lower bound loss"""
        e = e.detach().clone()

        betas_t = self.get_index_from_list(self.betas, t, x_t.shape)
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x_t.shape)
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        # Get predicted mean from model according to Equation 13 (pg 9)
        pred_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * e / sqrt_one_minus_alphas_cumprod_t
        )

        pred_log_variance = v * torch.log(betas_t) + (1 - v) * torch.log(posterior_variance_t)
        pred_variance = torch.exp(pred_log_variance)

        return pred_mean, pred_variance, pred_log_variance

    def _get_true_mean_var(self, x_0, x_t, t):
        mu_term1 = self.get_index_from_list(self.mu_term1, t, x_0.shape)
        mu_term2 = self.get_index_from_list(self.mu_term2, t, x_0.shape)

        true_mean = mu_term1 * x_0 + mu_term2 * x_t
        true_log_variance = self.get_index_from_list(self.posterior_log_variance_clipped, t, x_0.shape)
        true_variance = self.get_index_from_list(self.posterior_variance, t, x_0.shape)

        return true_mean, true_variance, true_log_variance

    def get_loss(self, x_0, t):
        """
        Perform forward step to get x_t and
        pass through model to approximate x_noisy

        BUG FIX:

            According to paper:
                L_simple = E [ || noise - noise_pred || ] ** 2 (pg 2)

            According to source https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils.py:
                L_simple = mean(mse(x))
        """
        B = x_0.size(0)

        x_t, noise = self.forward_step(x_0, t)
        e, v = self.model(x_t, t)

        # assert not torch.isnan(v).any()
        # assert not torch.isnan(e).any()

        # pred_mean, pred_var, pred_log_var = self._get_pred_mean_var(x_t, e, v, t)
        # true_mean, true_var, true_log_var = self._get_true_mean_var(x_0, x_t, t)

        # # simple loss term
        loss_simple = F.mse_loss(noise, e, reduction='mean')
        #
        # # variational lower bound loss term
        # assert not torch.isnan(true_mean).any()
        # assert not torch.isnan(true_log_var).any()
        # assert not torch.isnan(pred_mean).any()
        # assert not torch.isnan(pred_log_var).any()
        #
        # loss_vlb = self._normal_kl(true_mean, true_log_var, pred_mean, pred_log_var).mean()
        # print(loss_vlb)
        #
        # # hybrid loss
        # loss = loss_simple + self.lambda_ * loss_vlb

        return loss_simple

    def _normal_kl(self, mean1, logvar1, mean2, logvar2):
        print("---------")
        print(torch.exp(logvar1 - logvar2).mean())
        print(torch.exp(-logvar2).mean())

        return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) +
                      ((mean1 - mean2) ** 2) * torch.exp(-logvar2))

    def train_step(self, batch):
        B = batch.size(0)

        t = torch.randint(0, self.T, (B,), device=self.device).long()
        loss = self.get_loss(batch, t) / self.agg_grad
        loss.backward()

        if self.t % self.agg_grad == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
        noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        mean = sqrt_alphas_cumprod_t * x_0
        variance = sqrt_one_minus_alphas_cumprod_t * noise

        return mean + variance, noise

    def backward_step(self, x, t):
        """
        TODO: Math needs verification

        One denoising step using model mean prediction and precomputed variance
        e = noise with N(0, 1)

        model_mean = sqrt(1/alphas) * (x - betas * pred_e / sqrt(1 - alpha_prod))
        variance = sqrt(posterior_variance) * e

        reparameterization trick?
        x_t-1 = model_mean + variance
        """
        e, v = self.model(x, t)

        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x.shape)

        # Get predicted mean from model according to Equation 13 (pg 9)
        pred_mean = sqrt_recip_alphas_t * (
            x - betas_t * e / sqrt_one_minus_alphas_cumprod_t
        )

        if t == 0: return pred_mean

        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)

        return pred_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.inference_mode()
    def plot_denoising_process(self, num_images=10, step=1, save=None):
        """
        t = 0 is image
        t = 300 is noise
        """
        img = torch.randn((1, 3, self.img_size, self.img_size), device=self.device)
        step_size = int(self.T/num_images)

        for i in range(0, self.T, step)[::-1]:
            t = torch.full((1,), i, device=self.device, dtype=torch.long)
            img = self.backward_step(img, t)

            if i % step_size == 0:
                plt.subplot(1, num_images, int(i/step_size + 1))
                show_tensor_image(img.detach().cpu())

        show_tensor_image(img.detach().cpu(), save=save)
        plt.pause(0.01)

