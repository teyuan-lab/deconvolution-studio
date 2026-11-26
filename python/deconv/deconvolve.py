"""Non-blind deconvolution algorithms (PSF known or already estimated).

All functions accept a grayscale (H, W) or colour (H, W, C) float image in
[0, 1] and a normalised PSF.  Boundaries are handled by reflect padding +
edge tapering so the FFT's circular assumption does not cause ringing.

Algorithms
----------
inverse_filter      truncated inverse filter (didactic - shows noise blow-up)
wiener              Wiener filter, scalar noise-to-signal ratio
tikhonov            constrained least squares with a gradient (Laplacian) prior
richardson_lucy     Richardson-Lucy (Poisson ML), optional TV regularisation
tv_admm             total-variation regularised least squares solved by ADMM
"""
from __future__ import annotations

import numpy as np
from scipy import fft as _sfft

from .utils import edgetaper, estimate_noise_sigma, normalize_psf, prepare, psf2otf

def _fft2(a):
    return _sfft.fft2(a, workers=-1)


def _ifft2(a):
    return _sfft.ifft2(a, workers=-1)


METHODS = ("wiener", "tikhonov", "rl", "rl_tv", "tv", "inverse")

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _per_channel(fn=None, *, masked=False):
    """Decorator: apply a 2-D algorithm to each channel of a colour image,
    and take care of padding / cropping.

    masked=False : reflect-pad + edge-taper (approximate periodic boundary)
    masked=True  : the image is embedded in a larger canvas whose border
                   pixels are treated as *unknown*; the algorithm receives a
                   0/1 `mask` of observed pixels.  This removes boundary
                   ringing entirely (Almeida & Figueiredo 2013 /
                   Bertero & Boccacci 2005)."""
    if fn is None:
        return lambda f: _per_channel(f, masked=masked)

    def wrapper(image, psf, *args, **kwargs):
        image = np.asarray(image, dtype=np.float64)
        psf = normalize_psf(psf)
        if image.ndim == 3:
            return np.stack([wrapper(image[..., c], psf, *args, **kwargs)
                             for c in range(image.shape[2])], axis=-1)
        if masked and kwargs.get("boundary", "unknown") == "unknown":
            py, px = psf.shape
            H = _sfft.next_fast_len(image.shape[0] + 2 * py)
            W = _sfft.next_fast_len(image.shape[1] + 2 * px)
            prepped = np.pad(image, ((py, H - image.shape[0] - py),
                                     (px, W - image.shape[1] - px)), mode="reflect")
            sl = (slice(py, py + image.shape[0]), slice(px, px + image.shape[1]))
            mask = np.zeros_like(prepped)
            mask[sl] = 1.0
            kwargs["mask"] = mask
        else:
            prepped, sl = prepare(image, psf)
        kwargs.pop("boundary", None)
        cb = kwargs.get("callback")
        if cb is not None:      # give callbacks the un-padded image
            kwargs["callback"] = lambda k, x: cb(k, x[sl])
        out = fn(prepped, psf, *args, **kwargs)
        return out[sl]
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _grad_otfs(shape):
    """OTFs of forward-difference operators Dx, Dy."""
    dx = psf2otf(np.array([[1.0, -1.0]]), shape)
    dy = psf2otf(np.array([[1.0], [-1.0]]), shape)
    return dx, dy


def _grad(x):
    return np.roll(x, -1, axis=1) - x, np.roll(x, -1, axis=0) - x


def _div(px, py):
    """Negative adjoint of the forward gradient (periodic)."""
    return (px - np.roll(px, 1, axis=1)) + (py - np.roll(py, 1, axis=0))


def gcv_lambda(image, psf, regularizer: str = "gradient"):
    """Generalised cross-validation choice of the Tikhonov weight.

    For the Fourier-diagonal filter F = |H|^2 / (|H|^2 + lam |R|^2)
        GCV(lam) = ||(1 - F) Y||^2 / (sum(1 - F))^2
    is minimised over a log grid.  No noise estimate is needed and the
    choice adapts to both the image content and how ill-posed the PSF is.
    regularizer : 'gradient' (|R|^2 = |Dx|^2+|Dy|^2) or 'identity' (|R|^2 = 1)."""
    H2 = np.abs(psf2otf(psf, image.shape)) ** 2
    if regularizer == "gradient":
        Dx, Dy = _grad_otfs(image.shape)
        R2 = np.abs(Dx) ** 2 + np.abs(Dy) ** 2
    else:
        R2 = np.ones_like(H2)
    Y2 = np.abs(_fft2(image)) ** 2
    best_g, best_l = np.inf, 1e-3
    for lam in np.logspace(-6, 0, 49):
        one_minus_F = lam * R2 / (H2 + lam * R2)
        g = np.sum(one_minus_F ** 2 * Y2) / np.sum(one_minus_F) ** 2
        if g < best_g:
            best_g, best_l = g, lam
    return float(best_l)


# --------------------------------------------------------------------------
# Linear (closed form) filters
# --------------------------------------------------------------------------

@_per_channel
def inverse_filter(image, psf, threshold: float = 0.05):
    """Truncated inverse filter: X = Y / H where |H| > threshold, else 0.
    Included to demonstrate why regularisation is needed."""
    H = psf2otf(psf, image.shape)
    Y = _fft2(image)
    mask = np.abs(H) > threshold
    X = np.zeros_like(Y)
    X[mask] = Y[mask] / H[mask]
    return np.real(_ifft2(X))


@_per_channel
def wiener(image, psf, nsr: float | None = None):
    """Wiener filter  X = H* Y / (|H|^2 + NSR).

    nsr : noise-to-signal power ratio.  If None it is estimated as
          sigma_noise^2 / var(image) (with a small floor)."""
    if nsr is None:
        sigma = estimate_noise_sigma(image)
        nsr = max(60.0 * sigma ** 2, 1e-4)   # tuned on natural test images
    H = psf2otf(psf, image.shape)
    Y = _fft2(image)
    X = np.conj(H) * Y / (np.abs(H) ** 2 + nsr)
    return np.real(_ifft2(X))


@_per_channel
def tikhonov(image, psf, lam: float | None = None):
    """Constrained least squares / Tikhonov with a gradient prior:

        argmin_x ||h*x - y||^2 + lam ||grad x||^2
        X = H* Y / (|H|^2 + lam (|Dx|^2 + |Dy|^2))

    The gradient prior matches the ~1/f^2 power spectrum of natural images,
    so it suppresses high-frequency noise more than low frequencies.
    lam : if None, chosen by generalised cross-validation."""
    if lam is None:
        lam = gcv_lambda(image, psf, "gradient")
    H = psf2otf(psf, image.shape)
    Dx, Dy = _grad_otfs(image.shape)
    Y = _fft2(image)
    X = np.conj(H) * Y / (np.abs(H) ** 2 + lam * (np.abs(Dx) ** 2 + np.abs(Dy) ** 2))
    return np.real(_ifft2(X))


# --------------------------------------------------------------------------
# Iterative methods
# --------------------------------------------------------------------------

@_per_channel(masked=True)
def richardson_lucy(image, psf, iterations: int = 30, tv: float = 0.0,
                    eps: float = 1e-7, callback=None, mask=None):
    """Richardson-Lucy deconvolution (maximum likelihood for Poisson noise).

        x_{k+1} = x_k * H^T( y / (H x_k) )

    With tv > 0 this becomes RL-TV (Dey et al. 2006):

        x_{k+1} = x_k / (1 - tv * div(grad x_k / |grad x_k|)) * H^T( y / H x_k )

    which suppresses the noise amplification of plain RL at high iteration
    counts.  Typical tv: 0.001 - 0.01.

    boundary='unknown' (default) uses the masked RL update
        x_{k+1} = x_k / H^T(M) * H^T( M y / (H x_k) )
    so pixels outside the frame are estimated, not assumed.  Pass
    boundary='taper' for the classic reflect + edge-taper approach."""
    y = np.clip(image, 0, None) + eps
    H = psf2otf(psf, y.shape)
    Hc = np.conj(H)
    if mask is None:
        mask = np.ones_like(y)
    norm = np.real(_ifft2(_fft2(mask) * Hc))
    norm = np.maximum(norm, 1e-3)
    x = np.full_like(y, float(np.mean(y[mask > 0])))
    for k in range(iterations):
        Hx = np.real(_ifft2(_fft2(x) * H))
        ratio = mask * y / np.maximum(Hx, eps)
        corr = np.real(_ifft2(_fft2(ratio) * Hc)) / norm
        if tv > 0:
            gx, gy = _grad(x)
            mag = np.sqrt(gx ** 2 + gy ** 2 + 1e-6)
            denom = 1.0 - tv * _div(gx / mag, gy / mag)
            x = x * corr / np.maximum(denom, 0.1)
        else:
            x = x * corr
        x = np.maximum(x, 0)
        if callback is not None:
            callback(k, x)
    return x


@_per_channel(masked=True)
def tv_admm(image, psf, lam: float | None = None, rho: float | None = None,
            iterations: int = 80, mask=None, mu: float = 1.0):
    """Total-variation regularised deconvolution via ADMM:

        argmin_x  1/2 || M (h*x) - y ||^2 + lam * || grad x ||_{2,1}

    M masks the observed pixels: x lives on a canvas larger than the image
    by the PSF size, and pixels outside the frame are free (no boundary
    assumption -> no boundary ringing; Almeida & Figueiredo 2013).

    Splits v = h*x (data) and z = grad x (prior):
        x-step : (mu H^T H + rho D^T D) x = mu H^T (v - u1) + rho D^T (z - u2)  [FFT]
        v-step : v = (M y + mu (Hx + u1)) / (M + mu)                         [pixelwise]
        z-step : isotropic shrinkage of (grad x + u2) by lam / rho
    TV preserves sharp edges while removing noise; best for piecewise-smooth
    images.

    lam : if None, 0.25 x the GCV Tikhonov weight (empirically the TV optimum
          sits at 0.1-0.3x the Tikhonov optimum on natural images).
    rho : ADMM penalty for the gradient split, default max(10 lam, 0.05)."""
    if mask is None:
        mask = np.ones_like(image)
    if lam is None:
        lam = 0.25 * gcv_lambda(edgetaper(image, psf), psf, "gradient")
    if rho is None:
        rho = max(10 * lam, 0.05)
    H = psf2otf(psf, image.shape)
    Hc = np.conj(H)
    Dx, Dy = _grad_otfs(image.shape)
    denom = mu * np.abs(H) ** 2 + rho * (np.abs(Dx) ** 2 + np.abs(Dy) ** 2)
    My = mask * image
    x = image.copy()
    Hx = np.real(_ifft2(_fft2(x) * H))
    v = Hx.copy()
    u1 = np.zeros_like(x)
    zx = np.zeros_like(x); zy = np.zeros_like(x)
    ux = np.zeros_like(x); uy = np.zeros_like(x)
    for _ in range(iterations):
        rhs = mu * Hc * _fft2(v - u1) + rho * (np.conj(Dx) * _fft2(zx - ux)
                                                    + np.conj(Dy) * _fft2(zy - uy))
        x = np.real(_ifft2(rhs / denom))
        Hx = np.real(_ifft2(_fft2(x) * H))
        v = (My + mu * (Hx + u1)) / (mask + mu)
        gx, gy = _grad(x)
        vx, vy = gx + ux, gy + uy
        mag = np.sqrt(vx ** 2 + vy ** 2)
        shrink = np.maximum(1 - (lam / rho) / np.maximum(mag, 1e-12), 0)
        zx, zy = shrink * vx, shrink * vy
        u1 += Hx - v
        ux += gx - zx
        uy += gy - zy
    return x


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------

def deconvolve(image, psf, method: str = "tv", clip: bool = True, **params):
    """Run a named deconvolution method.

    method : 'wiener' | 'tikhonov' | 'rl' | 'rl_tv' | 'tv' | 'inverse'
    params : forwarded to the algorithm (nsr, lam, iterations, tv, ...)"""
    m = method.lower()
    if m == "wiener":
        out = wiener(image, psf, **params)
    elif m in ("tikhonov", "cls"):
        out = tikhonov(image, psf, **params)
    elif m in ("rl", "richardson_lucy"):
        out = richardson_lucy(image, psf, **params)
    elif m == "rl_tv":
        params.setdefault("tv", 0.003)
        params.setdefault("iterations", 50)
        out = richardson_lucy(image, psf, **params)
    elif m in ("tv", "tv_admm"):
        out = tv_admm(image, psf, **params)
    elif m == "inverse":
        out = inverse_filter(image, psf, **params)
    else:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    return np.clip(out, 0, 1) if clip else out
