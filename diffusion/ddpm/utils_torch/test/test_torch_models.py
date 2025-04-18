import pytest
import torch

from ..unet import Unet, SimpleUnet
from ..diffusion import sample


@pytest.mark.parametrize("b, c, d", [(64, 1, 28), (128, 3, 32), (256, 5, 28)])
def test_unet_forward(b: int, c: int, d: int):
    s = (b, c, d, d)

    unet = Unet(downs=[16, 32, 64], channels=c)
    t = torch.zeros(b)
    x = torch.zeros(s)

    z = unet(x, t)
    assert z.shape == s


@pytest.mark.parametrize("b, c, d", [(64, 1, 28), (128, 3, 32), (256, 5, 28)])
def test_simple_unet_forward(b: int, c: int, d: int):
    s = (b, c, d, d)

    unet = SimpleUnet(down_channels=[16, 32, 64], channels=c)

    t = torch.zeros(b)
    x = torch.zeros(s)

    z = unet(x, t)
    assert z.shape == s


@pytest.mark.parametrize("b, c, d", [(16, 1, 28), (32, 3, 32)])
def test_simple_unet_sampling(b: int, c: int, d: int):
    s = (b, c, d, d)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    unet = Unet(downs=[16, 32, 64], channels=c).to(device)
    T = 100  # faster sampling
    t = torch.ones(T).to(device) * 0.1

    x = sample(unet, s, T, t, t, t, t)

    assert len(x) == 1

    x = x[0]

    assert x.shape == s
