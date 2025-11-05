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

## POC object detection pipeline

Test `opencv` Python API, 
to build a video-streaming pipeline to detect objects in real time, with :

- *Mixture Of Gaussians* (MOG2) model for background substraction
- *Probabilistic Hough Lines Transform* for line detection

```bibtex
@article{zivkovic2006efficient,
  title={Efficient adaptive density estimation per image pixel for the task of background subtraction},
  author={Zivkovic, Zoran and Van Der Heijden, Ferdinand},
  journal={Pattern recognition letters},
  volume={27},
  number={7},
  pages={773--780},
  year={2006},
  publisher={Elsevier}
}
```

```bibtex
@article{kiryati1991probabilistic,
  title={A probabilistic Hough transform},
  author={Kiryati, Nahum and Eldar, Yuval and Bruckstein, Alfred M},
  journal={Pattern recognition},
  volume={24},
  number={4},
  pages={303--316},
  year={1991},
  publisher={Elsevier}
}
```

The final goal is to embed the model with :
<a href="https://github.com/anuraghazra/convoychat">
<img height=50 align="center" src="https://go-skill-icons.vercel.app/api/icons?i=raspberrypi,arduino" />
</a>

