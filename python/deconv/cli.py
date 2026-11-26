"""Command line interface.

    python -m deconv estimate   blurred.png --method auto -o psf.npy
    python -m deconv deconvolve blurred.png --psf psf.npy --method tv -o sharp.png
    python -m deconv deconvolve blurred.png --estimate auto -o sharp.png
    python -m deconv simulate   sharp.png --motion 15 30 --noise 0.01 -o blurred.png
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import psf as P
from .deconvolve import METHODS, deconvolve
from .utils import blur, load_image, load_psf, save_image


def _save_psf(psf, path):
    if path.lower().endswith(".npy"):
        np.save(path, psf)
    elif path.lower().endswith((".txt", ".csv")):
        np.savetxt(path, psf, delimiter="," if path.lower().endswith(".csv") else " ")
    else:
        save_image(path, psf / psf.max())


def _psf_from_args(a, img):
    if a.psf:
        return load_psf(a.psf), {"kind": "file", "path": a.psf}
    if a.gaussian is not None:
        return P.gaussian_psf(a.gaussian), {"kind": "gaussian", "sigma": a.gaussian}
    if a.disk is not None:
        return P.disk_psf(a.disk), {"kind": "disk", "radius": a.disk}
    if a.motion is not None:
        return P.motion_psf(*a.motion), {"kind": "motion", "length": a.motion[0], "angle": a.motion[1]}
    if a.estimate:
        kw = {"size": a.size} if a.estimate in ("blind", "point") else {}
        return P.estimate_psf(img, a.estimate, **kw)
    return None, None


def _add_psf_args(p):
    g = p.add_argument_group("PSF (choose one)")
    g.add_argument("--psf", help="PSF file (.npy/.txt/.csv/image)")
    g.add_argument("--gaussian", type=float, metavar="SIGMA")
    g.add_argument("--disk", type=float, metavar="RADIUS")
    g.add_argument("--motion", type=float, nargs=2, metavar=("LENGTH", "ANGLE"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="deconv", description="PSF estimation and image deconvolution")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("estimate", help="estimate the PSF of a blurred image")
    e.add_argument("image")
    e.add_argument("--method", default="auto",
                   choices=["auto", "gaussian", "defocus", "motion", "blind", "point", "edge"])
    e.add_argument("--size", type=int, default=25, help="kernel size for blind/point")
    e.add_argument("--roi", type=int, nargs=4, metavar=("ROW", "COL", "H", "W"),
                   help="edge region for --method edge")
    e.add_argument("-o", "--output", default="psf.npy", help=".npy/.txt/.csv or image")

    d = sub.add_parser("deconvolve", help="deblur an image")
    d.add_argument("image")
    _add_psf_args(d)
    d.add_argument("--estimate", choices=["auto", "gaussian", "defocus", "motion", "blind", "point", "edge"],
                   help="estimate the PSF from the image instead of supplying one")
    d.add_argument("--size", type=int, default=25)
    d.add_argument("--method", default="tv", choices=list(METHODS))
    d.add_argument("--nsr", type=float, help="Wiener noise-to-signal ratio")
    d.add_argument("--lam", type=float, help="regularisation weight (tikhonov / tv)")
    d.add_argument("--iterations", type=int, help="iterations (rl / rl_tv / tv)")
    d.add_argument("--tv", type=float, help="TV weight for rl_tv")
    d.add_argument("--save-psf", help="also save the PSF used")
    d.add_argument("-o", "--output", default="deconvolved.png")

    s = sub.add_parser("simulate", help="blur + add noise to a sharp image")
    s.add_argument("image")
    _add_psf_args(s)
    s.add_argument("--noise", type=float, default=0.0, help="Gaussian noise std (0-1 scale)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("-o", "--output", default="blurred.png")

    a = ap.parse_args(argv)

    if a.cmd == "estimate":
        img = load_image(a.image)
        kw = {}
        if a.method in ("blind", "point"):
            kw["size"] = a.size
        if a.method == "edge" and a.roi:
            kw["roi"] = tuple(a.roi)
        psf, info = P.estimate_psf(img, a.method, **kw)
        _save_psf(psf, a.output)
        print(json.dumps({k: v for k, v in info.items()}, default=float, indent=2))
        print(f"PSF {psf.shape} saved to {a.output}")
        return 0

    if a.cmd == "deconvolve":
        img = load_image(a.image)
        psf, info = _psf_from_args(a, img)
        if psf is None:
            print("error: give a PSF (--psf/--gaussian/--disk/--motion) or --estimate", file=sys.stderr)
            return 2
        print("PSF:", json.dumps(info, default=float))
        params = {k: getattr(a, k) for k in ("nsr", "lam", "iterations", "tv") if getattr(a, k) is not None}
        out = deconvolve(img, psf, a.method, **params)
        save_image(a.output, out)
        if a.save_psf:
            _save_psf(psf, a.save_psf)
        print(f"saved {a.output}")
        return 0

    if a.cmd == "simulate":
        img = load_image(a.image)
        psf, info = _psf_from_args(a, img)
        if psf is None:
            print("error: give a PSF", file=sys.stderr)
            return 2
        rng = np.random.default_rng(a.seed)
        out = blur(img, psf) + rng.normal(0, a.noise, img.shape)
        save_image(a.output, out)
        print(f"saved {a.output}  ({json.dumps(info, default=float)}, noise={a.noise})")
        return 0


if __name__ == "__main__":
    sys.exit(main())
