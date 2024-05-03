# Denoising Diffusion Probabilistic Models (DDPM) for image generation

A Generative Model that outperform GANs in terms of compute and benchmarks.

Predecessor of many state of the art generative models such as DALLE and Stable Diffusion.


## Results (100 epochs):


![alt text](https://github.com/YHL04/ddpm/blob/main/images/diffusionprocess.png)



## Improved DDPM Changes

- [X] Learning: Changing loss function and variance formula

- [X] Cosine Schedule

- [X] Faster Sampling by changing number of timesteps

- [X] Scalable transformer for diffusion

- [ ] Importance Sampling


## Citations

```
@article{ho2020denoising,
         title={Denoising Diffusion Probabilistic Models}, 
         author={Jonathan Ho and Ajay Jain and Pieter Abbeel},
         year={2020},
         eprint={2006.11239},
         archivePrefix={arXiv},
         primaryClass={cs.LG}
}
```

```bibtex
@article{nichol2021improved,
         title={Improved Denoising Diffusion Probabilistic Models}, 
         author={Alex Nichol and Prafulla Dhariwal},
         year={2021},
         eprint={2102.09672},
         archivePrefix={arXiv},
         primaryClass={cs.LG}
}
```

```
@article{peebles2023scalable,
         title={Scalable Diffusion Models with Transformers}, 
         author={William Peebles and Saining Xie},
         year={2023},
         eprint={2212.09748},
         archivePrefix={arXiv},
         primaryClass={cs.CV}
}
```