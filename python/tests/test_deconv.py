"""Quick self-test:  python -m pytest tests   (or  python tests/test_deconv.py)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import deconv as D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
X = D.load_image(os.path.join(HERE, "..", "cameraman.png"), gray=True)[128:384, 128:384]
RNG = np.random.default_rng(0)


def _blur(psf, s=0.005):
    return D.blur(X, psf) + RNG.normal(0, s, X.shape)


def test_every_method_improves_psnr():
    psf = D.gaussian_psf(2.0)
    y = _blur(psf)
    for m in ("wiener", "tikhonov", "rl", "rl_tv", "tv"):
        assert D.psnr(X, D.deconvolve(y, psf, m)) > D.psnr(X, y) + 1.0, m


def test_gaussian_blind():
    _, s = D.estimate_gaussian_blind(_blur(D.gaussian_psf(2.0)))
    assert abs(s - 2.0) < 0.3


def test_defocus_blind():
    _, r = D.estimate_defocus_blind(D.blur(D.load_image(os.path.join(HERE, "..", "cameraman.png"), gray=True),
                                           D.disk_psf(5)))
    assert abs(r - 5) < 0.5


def test_motion_blind():
    x = D.load_image(os.path.join(HERE, "..", "cameraman.png"), gray=True)
    _, L, a = D.estimate_motion_blind(D.blur(x, D.motion_psf(15, 30)) + RNG.normal(0, 0.005, x.shape))
    assert abs(L - 15) < 1.5 and min(abs(a - 30), 180 - abs(a - 30)) < 3


def test_edge():
    n = 160
    yy, xx = np.indices((n, n))
    chart = np.where((xx - n / 2) * np.cos(0.1) + (yy - n / 2) * np.sin(0.1) > 0, 0.8, 0.1)
    _, s = D.estimate_from_edge(D.blur(chart, D.gaussian_psf(1.5)))
    assert abs(s - 1.5) < 0.15


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
