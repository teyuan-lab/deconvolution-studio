"""Batch demo: reproduces every capability of the toolkit on synthetic data
where the ground truth is known, and writes figures + a results table.

    python demo.py            # full demo  (~1-2 min)
    python demo.py --quick    # smaller images

Output goes to ./demo_output/
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import deconv as D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "demo_output")
RNG = np.random.default_rng(2026)


def show(ax, img, title, cmap="gray"):
    ax.imshow(np.clip(img, 0, 1), cmap=cmap if img.ndim == 2 else None, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def show_psf(ax, psf, title, size=None):
    if size:
        p = np.zeros((size, size))
        h, w = psf.shape
        oy, ox = (size - h) // 2, (size - w) // 2
        if oy >= 0 and ox >= 0:
            p[oy:oy + h, ox:ox + w] = psf
        else:
            c = (h - size) // 2
            p = psf[c:c + size, c:c + size]
        psf = p
    ax.imshow(psf, cmap="inferno", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def score(ref, img):
    return D.psnr(ref, img), D.ssim(ref, img)


# --------------------------------------------------------------------------
def demo_algorithms(x, results):
    """1. Known PSF: compare every algorithm."""
    psf = D.gaussian_psf(2.5)
    y = D.blur(x, psf) + RNG.normal(0, 0.01, x.shape)
    entries = [("Original", x, None), ("Blurred + noise (sigma 2.5, 1%)", y, score(x, y))]
    for name, m, kw in [("Inverse filter (|H|>0.01)", "inverse", {"threshold": 0.01}),
                        ("Wiener", "wiener", {}),
                        ("Tikhonov / CLS", "tikhonov", {}),
                        ("Richardson-Lucy (30 it)", "rl", {"iterations": 30}),
                        ("RL-TV (50 it)", "rl_tv", {}),
                        ("TV-ADMM", "tv", {})]:
        t = time.time()
        o = D.deconvolve(y, psf, m, **kw)
        dt = time.time() - t
        s = score(x, o)
        entries.append((name, o, s))
        results["algorithms"].append({"method": name, "psnr": round(s[0], 2), "ssim": round(s[1], 3), "seconds": round(dt, 2)})
    fig, axs = plt.subplots(2, 4, figsize=(14, 8.2))
    for ax, (name, img, s) in zip(axs.ravel(), entries):
        show(ax, img, name + ("" if s is None else f"\nPSNR {s[0]:.2f} dB  SSIM {s[1]:.3f}"))
    fig.suptitle("1 - Non-blind deconvolution with a known Gaussian PSF", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "01_algorithms_known_psf.png"), dpi=110)
    plt.close(fig)


def demo_blind_parametric(xc, results):
    """2. Estimate the PSF blindly (auto family selection), then deblur."""
    cases = [("Gaussian sigma=2.0", D.gaussian_psf(2.0)),
             ("Defocus disk R=5", D.disk_psf(5)),
             ("Motion L=17, 35 deg", D.motion_psf(17, 35))]
    fig, axs = plt.subplots(3, 5, figsize=(16, 10))
    for row, (name, psf) in enumerate(cases):
        y = np.clip(D.blur(xc, psf) + RNG.normal(0, 0.005, xc.shape), 0, 1)
        t = time.time()
        est, info = D.estimate_psf(y, "auto")
        dt = time.time() - t
        o_est = D.deconvolve(y, est, "tv")
        o_true = D.deconvolve(y, psf, "tv")
        desc = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in info.items() if k != "scores"}
        results["blind_parametric"].append({
            "truth": name, "estimated": desc, "seconds": round(dt, 2),
            "psnr_blurred": round(D.psnr(xc, y), 2),
            "psnr_with_estimated_psf": round(D.psnr(xc, o_est), 2),
            "psnr_with_true_psf": round(D.psnr(xc, o_true), 2)})
        S = max(psf.shape[0], est.shape[0])
        est_txt = ", ".join(f"{k}={v}" for k, v in desc.items())
        show_psf(axs[row, 0], psf, f"True PSF\n{name}", S)
        show_psf(axs[row, 1], est, f"Estimated PSF\n{est_txt}", S)
        show(axs[row, 2], y, f"Blurred  PSNR {D.psnr(xc, y):.2f}")
        show(axs[row, 3], o_est, f"Deblurred (estimated PSF)\nPSNR {D.psnr(xc, o_est):.2f}")
        show(axs[row, 4], o_true, f"Deblurred (true PSF)\nPSNR {D.psnr(xc, o_true):.2f}")
    fig.suptitle("2 - Blind parametric PSF estimation (kind='auto') + TV deconvolution", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "02_blind_parametric.png"), dpi=100)
    plt.close(fig)


def demo_blind_kernel(x, results):
    """3. Non-parametric blind kernel estimation for an irregular shake."""
    # irregular camera-shake trajectory
    t = np.linspace(0, 1, 400)
    traj_x = 9 * np.sin(2.2 * np.pi * t) + 4 * t
    traj_y = 6 * np.cos(1.3 * np.pi * t) - 3
    k = np.zeros((25, 25))
    np.add.at(k, (np.clip(np.round(traj_y + 12).astype(int), 0, 24),
                  np.clip(np.round(traj_x + 12).astype(int) - 2, 0, 24)), 1)
    from scipy import ndimage
    psf = D.center_psf(ndimage.gaussian_filter(k, 0.5))
    y = D.blur(x, psf) + RNG.normal(0, 0.003, x.shape)
    t0 = time.time()
    est = D.blind_estimate(y, size=31)
    dt = time.time() - t0
    o_est = D.deconvolve(y, est, "tv")
    o_true = D.deconvolve(y, psf, "tv")
    results["blind_kernel"] = {"seconds": round(dt, 1), "psnr_blurred": round(D.psnr(x, y), 2),
                               "psnr_with_estimated_psf": round(D.psnr(x, o_est), 2),
                               "psnr_with_true_psf": round(D.psnr(x, o_true), 2)}
    fig, axs = plt.subplots(1, 5, figsize=(17, 4.2))
    show_psf(axs[0], psf, "True kernel (camera shake)", 31)
    show_psf(axs[1], est, f"Blind estimate ({dt:.0f} s)", 31)
    show(axs[2], y, f"Blurred  PSNR {D.psnr(x, y):.2f}")
    show(axs[3], o_est, f"Deblurred, estimated kernel\nPSNR {D.psnr(x, o_est):.2f}")
    show(axs[4], o_true, f"Deblurred, true kernel\nPSNR {D.psnr(x, o_true):.2f}")
    fig.suptitle("3 - Non-parametric blind kernel estimation (L0 latent prior, coarse-to-fine)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "03_blind_kernel.png"), dpi=100)
    plt.close(fig)


def demo_point_source(results):
    """4. PSF from stars: astronomical style image."""
    H = W = 320
    sky = np.zeros((H, W))
    for _ in range(30):
        r, c = RNG.integers(16, H - 16, 2)
        sky[r, c] += RNG.uniform(0.5, 3)
    # extended object (galaxy-like)
    yy, xx = np.indices(sky.shape)
    gal = 0.25 * np.exp(-(((xx - 200) * 0.9 + (yy - 120) * 0.4) ** 2 / 300 + ((yy - 120) * 0.9 - (xx - 200) * 0.4) ** 2 / 60))
    scene = np.clip(sky * 0.4 + gal, 0, 1)
    # asymmetric (comatic) PSF
    py, px = np.mgrid[-8:9, -8:9]
    psf = D.normalize_psf(np.exp(-(px ** 2 / (2 * 1.6 ** 2) + py ** 2 / (2 * 2.6 ** 2)))
                          + 0.35 * np.exp(-((px - 3.5) ** 2 + (py - 2.5) ** 2) / 3))
    img = D.blur(scene, psf, mode="constant") + 0.04 + RNG.normal(0, 0.002, scene.shape)
    est = D.estimate_from_point_source(img, size=17)
    o = D.deconvolve(img - 0.04, est, "rl", iterations=60)
    err = float(np.abs(est - psf).sum())
    results["point_source"] = {"psf_L1_error": round(err, 3), "n_stars_used": len(D.psf.find_point_sources(img, 17, 10))}
    fig, axs = plt.subplots(1, 5, figsize=(17, 4))
    show(axs[0], scene ** 0.5, "True scene (gamma 0.5)")
    show(axs[1], np.clip(img - 0.04, 0, 1) ** 0.5, "Observed (gamma 0.5)")
    show_psf(axs[2], psf, "True PSF (comatic)")
    show_psf(axs[3], est, f"PSF from stars (L1 err {err:.2f})")
    show(axs[4], np.clip(o, 0, 1) ** 0.5, "Richardson-Lucy 60 it (gamma 0.5)")
    fig.suptitle("4 - PSF measured from point sources, then Richardson-Lucy", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "04_point_source.png"), dpi=100)
    plt.close(fig)


def demo_edge(results):
    """5. Slanted-edge PSF measurement (lens test chart)."""
    n = 220
    yy, xx = np.indices((n, n))
    a = np.deg2rad(5)
    chart = np.where((xx - n / 2) * np.cos(a) + (yy - n / 2) * np.sin(a) > 0, 0.85, 0.12)
    rows = []
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for sigma, col in [(1.0, "C0"), (2.0, "C1"), (3.5, "C2")]:
        y = D.blur(chart, D.gaussian_psf(sigma)) + RNG.normal(0, 0.004, chart.shape)
        _, s_est, (dist, lsf) = D.estimate_from_edge(y, return_lsf=True)
        rows.append({"true_sigma": sigma, "estimated_sigma": round(s_est, 3)})
        axs[1].plot(dist, lsf / lsf.max(), col, lw=1.2, label=f"true sigma {sigma} -> est {s_est:.2f}")
        axs[1].plot(dist, np.exp(-dist ** 2 / (2 * s_est ** 2)), col, lw=0.8, ls=":")
        if sigma == 2.0:
            show(axs[0], y, "Blurred slanted edge (sigma 2)")
    axs[1].set_xlim(-12, 12)
    axs[1].set_title("Line spread functions (solid: measured, dotted: fitted Gaussian)", fontsize=9)
    axs[1].set_xlabel("distance from edge (px)")
    axs[1].legend(fontsize=8)
    t = [r["true_sigma"] for r in rows]
    e = [r["estimated_sigma"] for r in rows]
    axs[2].plot([0, 4], [0, 4], "k--", lw=0.8)
    axs[2].plot(t, e, "o", ms=8)
    axs[2].set_xlabel("true sigma")
    axs[2].set_ylabel("estimated sigma")
    axs[2].set_title("Edge-based sigma estimate", fontsize=9)
    fig.suptitle("5 - PSF from a straight edge (slanted-edge / ESF method)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "05_edge_method.png"), dpi=100)
    plt.close(fig)
    results["edge"] = rows


def demo_rl_iterations(x, results):
    """6. Semi-convergence of Richardson-Lucy and the effect of TV."""
    psf = D.disk_psf(4)
    y = np.clip(D.blur(x, psf) + RNG.normal(0, 0.01, x.shape), 0, 1)
    curves = {}
    for name, tv in [("RL", 0.0), ("RL-TV (0.003)", 0.003)]:
        ps = []

        def record(k, xx, ps=ps):
            if k % 5 == 4:
                ps.append(D.psnr(x, np.clip(xx, 0, 1)))
        D.richardson_lucy(y, psf, iterations=150, tv=tv, callback=record)
        curves[name] = ps
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, ps in curves.items():
        ax.plot(np.arange(5, 5 * len(ps) + 1, 5), ps, label=name)
    ax.set_xlabel("iteration")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Richardson-Lucy semi-convergence: plain RL amplifies noise\nafter its peak; TV regularisation stabilises it", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "06_rl_iterations.png"), dpi=100)
    plt.close(fig)
    results["rl_peak"] = {k: {"best_psnr": round(max(v), 2), "best_iter": 5 * (int(np.argmax(v)) + 1),
                              "psnr_at_150": round(v[-1], 2)} for k, v in curves.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    gray = D.load_image(os.path.join(HERE, "cameraman.png"), gray=True)
    color = D.load_image(os.path.join(HERE, "astronaut.png"))
    if a.quick:
        gray, color = gray[::2, ::2], color[::2, ::2]
    results = {"algorithms": [], "blind_parametric": []}
    steps = [("algorithms", lambda: demo_algorithms(gray, results)),
             ("blind parametric", lambda: demo_blind_parametric(color, results)),
             ("blind kernel", lambda: demo_blind_kernel(gray, results)),
             ("point source", lambda: demo_point_source(results)),
             ("edge", lambda: demo_edge(results)),
             ("RL iterations", lambda: demo_rl_iterations(gray, results))]
    for name, fn in steps:
        t = time.time()
        print(f"[demo] {name} ...", end=" ", flush=True)
        fn()
        print(f"{time.time() - t:.1f} s")
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nFigures written to {OUT}")


if __name__ == "__main__":
    main()
