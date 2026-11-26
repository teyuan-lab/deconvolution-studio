"""Interactive demo (matplotlib window - no extra dependencies).

    python gui_demo.py                 # starts with the bundled cameraman image
    python gui_demo.py my_photo.jpg    # start with your own image

Two ways to use it
------------------
1. Simulation mode (default): pick a blur type and strength, press
   "Simulate blur", then "Deblur".  PSNR/SSIM against the sharp original
   are reported, so you can compare PSF estimators and algorithms.
2. Real image mode: open a blurred photo, choose "Auto estimate" or
   "Blind kernel" (or set the PSF by hand with the sliders) and "Deblur".

Controls
--------
Blur type + sliders : the PSF used for simulation, and the manual PSF.
PSF for deblurring  : Known/manual (sliders) | Auto estimate | Blind kernel
Algorithm           : Wiener | Tikhonov | Richardson-Lucy | RL-TV | TV-ADMM
Regularisation x    : multiplies the automatically chosen weight (log scale)
Iterations          : for RL / RL-TV
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider

import deconv as D
from deconv.deconvolve import gcv_lambda
from deconv.utils import edgetaper, estimate_noise_sigma, to_gray

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_DIM = 1024          # larger images are downsized to keep the GUI responsive


class App:
    def __init__(self, path=None):
        self.sharp = None        # ground truth (simulation mode only)
        self.observed = None     # image to deblur
        self.result = None
        self.psf_used = None
        self.fig = plt.figure(figsize=(15, 8.6))
        self.fig.canvas.manager.set_window_title("Deconvolution demo")
        self._build_axes()
        self.load(path or os.path.join(HERE, "cameraman.png"), as_sharp=True)
        self.simulate(None)

    # ------------------------------------------------------------------ UI
    def _build_axes(self):
        f = self.fig
        self.ax_in = f.add_axes([0.02, 0.40, 0.31, 0.55])
        self.ax_psf = f.add_axes([0.355, 0.55, 0.13, 0.40])
        self.ax_out = f.add_axes([0.51, 0.40, 0.31, 0.55])
        for a in (self.ax_in, self.ax_psf, self.ax_out):
            a.axis("off")
        self.ax_status = f.add_axes([0.355, 0.40, 0.14, 0.13])
        self.ax_status.axis("off")
        self.status = self.ax_status.text(0, 1, "", va="top", fontsize=8, family="monospace")

        f.text(0.02, 0.355, "Blur type (simulation / manual PSF)", fontsize=9, weight="bold")
        self.r_blur = RadioButtons(f.add_axes([0.02, 0.21, 0.15, 0.13]),
                                   ("Gaussian", "Defocus disk", "Motion"))
        self.s_size = Slider(f.add_axes([0.24, 0.31, 0.22, 0.03]), "sigma/R/len", 0.5, 30, valinit=2.5)
        self.s_angle = Slider(f.add_axes([0.24, 0.26, 0.22, 0.03]), "angle (deg)", 0, 180, valinit=30)
        self.s_noise = Slider(f.add_axes([0.24, 0.21, 0.22, 0.03]), "noise %", 0, 5, valinit=0.5)

        f.text(0.50, 0.355, "PSF for deblurring", fontsize=9, weight="bold")
        self.r_psf = RadioButtons(f.add_axes([0.50, 0.21, 0.14, 0.13]),
                                  ("Known / manual", "Auto estimate", "Blind kernel"))
        f.text(0.66, 0.355, "Algorithm", fontsize=9, weight="bold")
        self.r_alg = RadioButtons(f.add_axes([0.66, 0.17, 0.14, 0.17]),
                                  ("Wiener", "Tikhonov", "Richardson-Lucy", "RL-TV", "TV-ADMM"), active=4)
        self.s_reg = Slider(f.add_axes([0.24, 0.14, 0.22, 0.03]), "regularisation x (log10)", -2, 2, valinit=0)
        self.s_iter = Slider(f.add_axes([0.24, 0.09, 0.22, 0.03]), "RL iterations", 5, 200, valinit=30, valstep=5)

        self.b_open = Button(f.add_axes([0.84, 0.86, 0.14, 0.06]), "Open image...")
        self.b_sim = Button(f.add_axes([0.84, 0.78, 0.14, 0.06]), "Simulate blur")
        self.b_real = Button(f.add_axes([0.84, 0.70, 0.14, 0.06]), "Use image as-is\n(real blurred photo)")
        self.b_run = Button(f.add_axes([0.84, 0.58, 0.14, 0.08]), "DEBLUR", color="#c8e6c9", hovercolor="#a5d6a7")
        self.b_save = Button(f.add_axes([0.84, 0.48, 0.14, 0.06]), "Save result...")
        self.b_open.on_clicked(self.open_dialog)
        self.b_sim.on_clicked(self.simulate)
        self.b_real.on_clicked(self.use_as_is)
        self.b_run.on_clicked(self.run)
        self.b_save.on_clicked(self.save_dialog)
        f.text(0.84, 0.40, "Tip: in simulation mode the\nPSNR/SSIM vs. the sharp\noriginal is shown.",
               fontsize=8, color="0.35")

    def say(self, msg):
        self.status.set_text(msg)
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

    # ----------------------------------------------------------- image I/O
    def load(self, path, as_sharp=True):
        img = D.load_image(path)
        if img.ndim == 3 and np.allclose(img[..., 0], img[..., 1]) and np.allclose(img[..., 1], img[..., 2]):
            img = img[..., 0]
        m = max(img.shape[:2])
        if m > MAX_DIM:
            from scipy import ndimage
            s = MAX_DIM / m
            zoom = (s, s) if img.ndim == 2 else (s, s, 1)
            img = np.clip(ndimage.zoom(img, zoom, order=1), 0, 1)
        self.path = path
        if as_sharp:
            self.sharp = img
        self.observed = img
        self.result = None
        self.show()

    def open_dialog(self, _):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            p = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All", "*.*")])
            root.destroy()
        except Exception as e:  # pragma: no cover
            self.say(f"file dialog unavailable:\n{e}\nPass the path on the\ncommand line instead.")
            return
        if p:
            self.load(p, as_sharp=True)
            self.say(f"Loaded {os.path.basename(p)}\n{self.observed.shape}\nSimulate blur, or\n'Use image as-is'.")

    def save_dialog(self, _):
        if self.result is None:
            self.say("Nothing to save yet.")
            return
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
            root.destroy()
        except Exception:
            p = os.path.join(HERE, "gui_result.png")
        if p:
            D.save_image(p, self.result)
            np.save(os.path.splitext(p)[0] + "_psf.npy", self.psf_used)
            self.say(f"Saved\n{os.path.basename(p)}\n(+ _psf.npy)")

    # --------------------------------------------------------------- logic
    def manual_psf(self):
        kind = self.r_blur.value_selected
        v = self.s_size.val
        if kind == "Gaussian":
            return D.gaussian_psf(v), f"Gaussian sigma={v:.2f}"
        if kind == "Defocus disk":
            return D.disk_psf(v), f"Disk R={v:.2f}"
        return D.motion_psf(v, self.s_angle.val), f"Motion L={v:.1f} a={self.s_angle.val:.0f}"

    def simulate(self, _):
        if self.sharp is None:
            return
        psf, desc = self.manual_psf()
        rng = np.random.default_rng(0)
        y = D.blur(self.sharp, psf) + rng.normal(0, self.s_noise.val / 100, self.sharp.shape)
        self.observed = np.clip(y, 0, 1)
        self.true_psf = psf
        self.result = None
        self.psf_used = psf
        self.mode = "sim"
        self.show()
        self.say(f"Simulated: {desc}\nnoise {self.s_noise.val:.2f}%\nPSNR blurred "
                 f"{D.psnr(self.sharp, self.observed):.2f} dB\nNow press DEBLUR.")

    def use_as_is(self, _):
        self.observed = self.sharp
        self.mode = "real"
        self.result = None
        self.psf_used = None
        self.show()
        self.say("Real-image mode: the\nloaded image is deblurred.\nChoose 'Auto estimate'\nor 'Blind kernel'.")

    def run(self, _):
        t0 = time.time()
        y = self.observed
        src = self.r_psf.value_selected
        if src == "Known / manual":
            psf, desc = self.manual_psf()
        elif src == "Auto estimate":
            self.say("Estimating PSF (auto)...")
            psf, info = D.estimate_psf(y, "auto")
            desc = "\n  ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in info.items() if k != "scores")
        else:
            size = int(max(15, 2 * self.s_size.val + 9)) | 1
            self.say(f"Blind kernel estimation\n(size {size}) - a few seconds...")
            psf = D.blind_estimate(y, size=size)
            desc = f"blind kernel {psf.shape[0]}x{psf.shape[1]}"
        self.psf_used = psf
        alg = self.r_alg.value_selected
        mult = 10 ** self.s_reg.val
        g = to_gray(y)
        self.say(f"PSF: {desc}\nRunning {alg}...")
        if alg == "Wiener":
            nsr = max(60 * estimate_noise_sigma(g) ** 2, 1e-4) * mult
            out = D.deconvolve(y, psf, "wiener", nsr=nsr); par = f"nsr={nsr:.2e}"
        elif alg == "Tikhonov":
            lam = gcv_lambda(edgetaper(g, psf), psf) * mult
            out = D.deconvolve(y, psf, "tikhonov", lam=lam); par = f"lam={lam:.2e}"
        elif alg == "Richardson-Lucy":
            it = int(self.s_iter.val)
            out = D.deconvolve(y, psf, "rl", iterations=it); par = f"iter={it}"
        elif alg == "RL-TV":
            it = int(self.s_iter.val)
            out = D.deconvolve(y, psf, "rl_tv", iterations=it, tv=0.003 * mult); par = f"iter={it} tv={0.003 * mult:.1e}"
        else:
            lam = 0.25 * gcv_lambda(edgetaper(g, psf), psf) * mult
            out = D.deconvolve(y, psf, "tv", lam=lam); par = f"lam={lam:.2e}"
        self.result = out
        dt = time.time() - t0
        msg = f"PSF: {desc}\n{alg}: {par}\ntime {dt:.1f} s"
        if getattr(self, "mode", "sim") == "sim" and self.sharp is not None:
            msg += (f"\nPSNR {D.psnr(self.sharp, y):.2f} -> {D.psnr(self.sharp, out):.2f} dB"
                    f"\nSSIM {D.ssim(self.sharp, y):.3f} -> {D.ssim(self.sharp, out):.3f}")
        self.show()
        self.say(msg)

    # ------------------------------------------------------------- drawing
    def show(self):
        def draw(ax, img, title):
            ax.clear(); ax.axis("off")
            if img is not None:
                ax.imshow(np.clip(img, 0, 1), cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=1)
            ax.set_title(title, fontsize=10)
        draw(self.ax_in, self.observed, "Input (blurred)")
        draw(self.ax_out, self.result, "Deblurred" if self.result is not None else "Deblurred (press DEBLUR)")
        self.ax_psf.clear(); self.ax_psf.axis("off")
        if self.psf_used is not None:
            self.ax_psf.imshow(self.psf_used, cmap="inferno", interpolation="nearest")
        self.ax_psf.set_title("PSF", fontsize=10)
        self.fig.canvas.draw_idle()


def main():
    App(sys.argv[1] if len(sys.argv) > 1 else None)
    plt.show()


if __name__ == "__main__":
    main()
