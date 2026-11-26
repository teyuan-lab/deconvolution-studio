"""Shared helpers: FFT convolution, PSF<->OTF, boundary handling, noise
estimation, image I/O and quality metrics.

Only numpy / scipy / matplotlib (+ Pillow, which matplotlib ships with)
are required.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

# --------------------------------------------------------------------------
# PSF / OTF
# --------------------------------------------------------------------------

def normalize_psf(psf: np.ndarray) -> np.ndarray:
    """Clip negatives and scale the PSF so it sums to 1 (energy preserving)."""
    psf = np.clip(np.asarray(psf, dtype=np.float64), 0, None)
    s = psf.sum()
    if s <= 0:
        raise ValueError("PSF has no positive energy")
    return psf / s


def psf2otf(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Zero-pad a PSF to `shape` and circularly shift its centre to (0, 0)
    before taking the FFT (equivalent to MATLAB's psf2otf)."""
    psf = np.asarray(psf, dtype=np.float64)
    pad = np.zeros(shape, dtype=np.float64)
    ph, pw = psf.shape
    pad[:ph, :pw] = psf
    pad = np.roll(pad, (-(ph // 2), -(pw // 2)), axis=(0, 1))
    return np.fft.fft2(pad)


def otf2psf(otf: np.ndarray, psf_shape: tuple[int, int]) -> np.ndarray:
    """Inverse of :func:`psf2otf` - crop the centred PSF of size psf_shape."""
    psf = np.real(np.fft.ifft2(otf))
    ph, pw = psf_shape
    psf = np.roll(psf, (ph // 2, pw // 2), axis=(0, 1))
    return psf[:ph, :pw]


def fft_convolve(img: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Circular convolution of a 2-D image with a PSF via FFT."""
    otf = psf2otf(psf, img.shape)
    return np.real(np.fft.ifft2(np.fft.fft2(img) * otf))


def blur(img: np.ndarray, psf: np.ndarray, mode: str = "reflect") -> np.ndarray:
    """Physically realistic blur (non-circular) for simulation."""
    if img.ndim == 3:
        return np.stack([blur(img[..., c], psf, mode) for c in range(img.shape[2])], -1)
    # ndimage.convolve flips the kernel -> true convolution
    return ndimage.convolve(img, psf, mode=mode)


# --------------------------------------------------------------------------
# Boundary handling
# --------------------------------------------------------------------------

def edgetaper(img: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Blend the image borders with a blurred copy so the image becomes
    (approximately) periodic.  Reduces ringing of FFT based deconvolution.
    Port of the idea behind MATLAB's edgetaper."""
    blurred = np.real(np.fft.ifft2(np.fft.fft2(img) * psf2otf(psf, img.shape)))
    alpha = np.ones_like(img)
    for axis, n in enumerate(img.shape):
        w = max(2, min(psf.shape[axis], n // 4))       # taper width ~ PSF size
        ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(w) + 0.5) / w)  # raised cosine
        prof = np.ones(n)
        prof[:w] = ramp
        prof[n - w:] = ramp[::-1]
        shape = [1, 1]
        shape[axis] = n
        alpha = alpha * prof.reshape(shape)
    return alpha * img + (1 - alpha) * blurred


def pad_for_fft(img: np.ndarray, psf_shape: tuple[int, int]):
    """Reflect-pad the image by the PSF size and taper the outer border so
    circular FFT methods behave like linear convolution.  Returns the padded
    image and the slice that recovers the original region."""
    from scipy.fft import next_fast_len
    py, px = (int(np.ceil(s)) for s in psf_shape)
    H = next_fast_len(img.shape[0] + 2 * py)
    W = next_fast_len(img.shape[1] + 2 * px)
    padded = np.pad(img, ((py, H - img.shape[0] - py), (px, W - img.shape[1] - px)),
                    mode="reflect")
    sl = (slice(py, py + img.shape[0]), slice(px, px + img.shape[1]))
    return padded, sl


def prepare(img: np.ndarray, psf: np.ndarray):
    """pad + edgetaper; returns (prepared image, crop slice)."""
    padded, sl = pad_for_fft(img, psf.shape)
    return edgetaper(padded, psf), sl


# --------------------------------------------------------------------------
# Noise estimation
# --------------------------------------------------------------------------

def estimate_noise_sigma(img: np.ndarray) -> float:
    """Robust Gaussian noise std estimate from the finest Haar diagonal
    wavelet band (Donoho's MAD estimator)."""
    if img.ndim == 3:
        return float(np.mean([estimate_noise_sigma(img[..., c]) for c in range(img.shape[2])]))
    h, w = (img.shape[0] // 2) * 2, (img.shape[1] // 2) * 2
    x = img[:h, :w]
    hh = (x[0::2, 0::2] - x[0::2, 1::2] - x[1::2, 0::2] + x[1::2, 1::2]) / 2.0
    return float(np.median(np.abs(hh)) / 0.6745)


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return img[..., :3] @ np.array([0.299, 0.587, 0.114])


def load_image(path: str, gray: bool = False) -> np.ndarray:
    """Load an image as float64 in [0, 1]."""
    from PIL import Image
    im = Image.open(path)
    if im.mode in ("I;16", "I;16B", "I"):
        arr = np.asarray(im, dtype=np.float64)
        arr = arr / (65535.0 if arr.max() > 255 else 255.0)
    else:
        im = im.convert("RGB") if im.mode not in ("L", "RGB") else im
        arr = np.asarray(im, dtype=np.float64) / 255.0
    if gray:
        arr = to_gray(arr)
    return arr


def save_image(path: str, img: np.ndarray) -> None:
    from PIL import Image
    arr = np.clip(img, 0, 1)
    Image.fromarray((arr * 255 + 0.5).astype(np.uint8)).save(path)


def load_psf(path: str) -> np.ndarray:
    """Load a PSF from .npy / .txt / .csv or an image file."""
    p = path.lower()
    if p.endswith(".npy"):
        psf = np.load(path)
    elif p.endswith((".txt", ".csv")):
        psf = np.loadtxt(path, delimiter="," if p.endswith(".csv") else None)
    else:
        psf = load_image(path, gray=True)
    return normalize_psf(psf)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def psnr(ref: np.ndarray, img: np.ndarray, data_range: float = 1.0) -> float:
    mse = np.mean((np.asarray(ref, float) - np.asarray(img, float)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(data_range ** 2 / mse))


def ssim(ref: np.ndarray, img: np.ndarray, data_range: float = 1.0) -> float:
    """Gaussian-window SSIM (Wang et al. 2004), grayscale or mean over channels."""
    if ref.ndim == 3:
        return float(np.mean([ssim(ref[..., c], img[..., c], data_range) for c in range(ref.shape[2])]))
    x, y = np.asarray(ref, float), np.asarray(img, float)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    f = lambda a: ndimage.gaussian_filter(a, 1.5, truncate=3.5)
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx * mx, f(y * y) - my * my, f(x * y) - mx * my
    m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return float(m[5:-5, 5:-5].mean())
