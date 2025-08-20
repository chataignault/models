# Personnal model implementations catalogue

**Store implemented models, test scripts and ideas.**

Project structure :

```bash
├── computer_vision
│   ├── inception
│   └── yolox
├── diffusion
│   ├── dae
│   ├── ddpm
│   ├── ppca
│   └── vae
├── fixed_income
├── mastery
│   └── Harmonic_Analysis_on_Paths_Spaces.pdf
├── numerical_exercises
│   ├── dealing_cards
│   ├── discrete_execution_optimisation
│   ├── positive_path_count
│   └── proba_negative_asset
├── pdevnet
└── utils
    └── polars_hdf5
```

## Diffusion

### Implement Denoising Diffusion Probabilitic Model (DDPM)
- jax and pytorch implementation with variants
- get familiar with `lightning`, `tensorboard`
- [Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." Advances in neural information processing systems 33 (2020): 6840-6851.](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)

### Variational Auto-Encoder
- Sparse and convolutional variants
- Bernoulli or Gaussian distributions
- get familiar with `mlflow`, `pytorch-lion`

### Probabilistic PCA
- Link with sample generation
- Linear algebra numerical implementation : SVD, bidiagonalisation
- get familiar with `pytest`
- compare with the EM training

### Test with other celebrated models
- [Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)
    - HuggingFace reference : https://huggingface.co/stabilityai/stable-diffusion-2-1
- [Guo, Yuwei, et al. "Animatediff: Animate your personalized text-to-image diffusion models without specific tuning." arXiv preprint arXiv:2307.04725 (2023).](https://arxiv.org/pdf/2307.04725) 
    - HuggingFace reference : https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-2

## Computer Vision

### Inception block POC
Implement minimal inception network applied to CIFAR 10 classification.

### YOLOX inference
Test YOLOX algorithm for bounding box detection.

## Path Development
- Leverage path development inside models
- Application to time-series classification

## Mastery
- Litterature review on path developments 

- [ ] Add example scripts

## Fixed Income 
- Validate intuition on the difference between **PV01 and DV01** 

## Numerical implementation
Implement in Rust numerical verification of math questions.

- Counting discrete positive paths starting and ending at zero,
- Probability of a random walk to reach zero under time constraint,
- Optimal strategy to minimize a capped uniform law sequentially under constraints.

*** 

## References 
- https://documentation.ubuntu.com/wsl/en/latest/howto/gpu-cuda/

