

import torch
import torch.nn.functional as F
from torch.optim import Adam

import math
import matplotlib.pyplot as plt

from utils import show_tensor_image
from losses import normal_kl, discretized_gaussian_log_likelihood


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
        self.betas = cosine_beta_schedule(timesteps=T, device=device)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

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

    def get_loss(self, x_0, c, t):
        """
        Perform forward step to get x_t and
        pass through model to approximate x_noisy

        Args:
            x_0: original image
            c: classes of the images
            t: timestep
        """
        B = x_0.size(0)

        x_t, noise = self.forward_step(x_0, t)
        e, v = self.model(x_t, c, t)

        loss_simple = self.get_simple(noise, e)
        loss_vb = self.get_vb(e, v, x_0, x_t, t)

        return loss_simple + self.lambda_ * loss_vb

    def get_simple(self, noise, e, reduction='mean'):
        return F.mse_loss(noise, e, reduction=reduction)

    def get_vb(self, e, v, x_0, x_t, t):
        e = e.detach()

        x_recon = self.predict_start_from_noise(x_t, t=t, noise=e)
        x_recon.clamp_(-1., 1.)

        pred_mean, _, _ = self.q_posterior(x_start=x_recon, x_t=x_t, t=t)
        pred_log_variance = self.model_v_to_log_variance(v, t)
        pred_variance = pred_log_variance.exp()

        true_mean, true_variance, true_log_variance = self.q_posterior(x_start=x_0, x_t=x_t, t=t)

        kl = normal_kl(true_mean, true_log_variance, pred_mean, pred_log_variance)
        kl = kl.mean(dim=list(range(1, len(kl.shape)))) / torch.log(torch.tensor(2.0))

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_0, means=pred_mean, log_scales=0.5 * pred_log_variance
        )

        decoder_nll = decoder_nll.mean(dim=list(range(1, len(decoder_nll.shape)))) / torch.log(torch.tensor(2.0))

        output = torch.where((t==0), decoder_nll, kl)
        return output.mean()

    def train_step(self, batch):
        img, cls = batch
        B = img.size(0)

        t = torch.randint(0, self.T, (B,), device=self.device).long()
        loss = self.get_loss(img, cls, t) / self.agg_grad
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

    def backward_step(self, x, y, t):
        """
        TODO: Math needs verification

        One denoising step using model mean prediction and precomputed variance
        e = noise with N(0, 1)

        model_mean = sqrt(1/alphas) * (x - betas * pred_e / sqrt(1 - alpha_prod))
        variance = sqrt(posterior_variance) * e

        reparameterization trick?
        x_t-1 = model_mean + variance
        """
        mean, variance, log_variance = self.p_mean_variance(x, y, t)

        noise = torch.randn_like(x)
        return mean + (0.5 * log_variance).exp() * noise

    def predict_start_from_noise(self, x_t, t, noise):
        sqrt_recip_alphas_cumprod = self.get_index_from_list(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod = self.get_index_from_list(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

        return sqrt_recip_alphas_cumprod * x_t - sqrt_recipm1_alphas_cumprod * noise

    def q_posterior(self, x_start, x_t, t):
        mu_term1 = self.get_index_from_list(self.mu_term1, t, x_t.shape)
        mu_term2 = self.get_index_from_list(self.mu_term2, t, x_t.shape)

        posterior_mean = mu_term1 * x_start + mu_term2 * x_t
        posterior_variance = self.get_index_from_list(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self.get_index_from_list(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, y, t, clip_denoised=True):
        e, v = self.model(x, y, t)

        # get x_0 from predicted noise
        x_recon = self.predict_start_from_noise(x, t=t, noise=e)
        x_recon.clamp_(-1., 1.)

        # get log variance
        log_variance = self.model_v_to_log_variance(v, t)
        variance = log_variance.exp()

        model_mean, _, _ = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, variance, log_variance

    def model_v_to_log_variance(self, v, t):
        min_log = self.get_index_from_list(self.posterior_log_variance_clipped, t, v.shape)
        max_log = self.get_index_from_list(self.betas, t, v.shape).log()

        frac = (v + 1) / 2
        return frac * max_log + (1 - frac) * min_log

    @torch.inference_mode()
    def plot_denoising_process(self, num_batch=5, num_images=10, step=1, save=None):
        """
        t = 0 is image
        """
        img = torch.randn((num_batch, 3, self.img_size, self.img_size), device=self.device)
        cls = torch.randint(low=0, high=100, size=(num_batch,), dtype=torch.int64, device=self.device)
        step_size = int(self.T/num_images)

        for i in range(0, self.T, step)[::-1]:
            t = torch.full((num_batch,), i, device=self.device, dtype=torch.long)
            img = self.backward_step(img, cls, t)

            if i % step_size == 0:
                for n in range(num_batch):
                    # plt.subplot(num_batch, num_images, (n+1, int(i/step_size + 1)))
                    plt.subplot(num_batch, num_images, (n*num_images + int(i / step_size + 1)))
                    show_tensor_image(img[n].detach().cpu())

        show_tensor_image(img[n].detach().cpu(), save=save)
        plt.pause(0.01)

