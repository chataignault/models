# Computer Vision models

## Inception block POC

Implement and test the Inception block with dimensionality reduction.

Compare implementations training and inference in :
- `torch`
- `jax`, `flax`
- `burn` (Rust)
with CIFAR10 images.

```bibtex
@inproceedings{szegedy2015going,
  title={Going deeper with convolutions},
  author={Szegedy, Christian and Liu, Wei and Jia, Yangqing and Sermanet, Pierre and Reed, Scott and Anguelov, Dragomir and Erhan, Dumitru and Vanhoucke, Vincent and Rabinovich, Andrew},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1--9},
  year={2015}
}
```

## Pretrained Yolox for bounding box segmentation

Test the `burn` implementation of YOLOX.

> Maybe fixing minor mistake on model spec

```bibtex
@article{diwan2023object,
  title={Object detection using YOLO: challenges, architectural successors, datasets and applications},
  author={Diwan, Tausif and Anirudh, G and Tembhurne, Jitendra V},
  journal={multimedia Tools and Applications},
  volume={82},
  number={6},
  pages={9243--9275},
  year={2023},
  publisher={Springer}
}
```


