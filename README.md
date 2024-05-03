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

```bibtex
@article{ho2020denoising,
  title={Denoising diffusion probabilistic models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={6840--6851},
  year={2020}
}
```

```bibtex
@inproceedings{nichol2021improved,
  title={Improved denoising diffusion probabilistic models},
  author={Nichol, Alexander Quinn and Dhariwal, Prafulla},
  booktitle={International conference on machine learning},
  pages={8162--8171},
  year={2021},
  organization={PMLR}
}
```

```bibtex
@inproceedings{peebles2023scalable,
  title={Scalable diffusion models with transformers},
  author={Peebles, William and Xie, Saining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4195--4205},
  year={2023}
}
```