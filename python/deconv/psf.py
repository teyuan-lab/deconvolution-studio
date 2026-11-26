"""Point-spread-function models and estimators.

Models
------
gaussian_psf(sigma)              optical / atmospheric / sensor blur
disk_psf(radius)                 defocus (geometric optics: uniform disk)
motion_psf(length, angle)        linear camera / object motion

Estimators
----------
estimate_from_point_source(img)  crop + register + stack isolated bright points
estimate_from_edge(img)          slanted-edge ESF -> LSF -> Gaussian sigma
estimate_gaussian_blind(img)     fit the radial power spectrum (no target needed)
estimate_defocus_blind(img)      match spectrum zeros of a disk OTF
estimate_motion_blind(img)       spectrum stripe angle + cepstrum length
blind_estimate(img, size)        non-parametric kernel by multi-scale
                                 alternating minimisation (Cho & Lee style)
estimate_psf(img, kind=...)      dispatcher, incl. kind='auto'
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, optimize
from scipy import fft as _sfft

from .utils import normalize_psf, psf2otf, to_gray

# ==========================================================================
# PSF models
# ==========================================================================

def _odd(n: float) -> int:
    n = int(np.ceil(n))
    return n if n % 2 == 1 else n + 1


def gaussian_psf(sigma: float, size: int | None = None) -> np.ndarray:
    """Isotropic 2-D Gaussian PSF.  size defaults to ~ 6.5 sigma (odd)."""
    sigma = max(float(sigma), 1e-3)
    size = size or max(3, _odd(6.5 * sigma))
    r = (size - 1) / 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return normalize_psf(np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)))


def disk_psf(radius: float, size: int | None = None, oversample: int = 8) -> np.ndarray:
    """Uniform (pillbox) defocus PSF with anti-aliased rim."""
    radius = max(float(radius), 0.5)
    size = size or _odd(2 * radius + 3)
    r = (size - 1) / 2
    n = size * oversample
    c = (np.arange(n) + 0.5) / oversample - 0.5 - r
    yy, xx = np.meshgrid(c, c, indexing="ij")
    hi = (xx ** 2 + yy ** 2 <= radius ** 2).astype(float)
    psf = hi.reshape(size, oversample, size, oversample).mean(axis=(1, 3))
    return normalize_psf(psf)


def motion_psf(length: float, angle: float = 0.0, size: int | None = None,
               oversample: int = 8) -> np.ndarray:
    """Linear motion blur of `length` pixels.  `angle` in degrees, measured
    counter-clockwise from the +x axis as seen on screen (row index grows
    downward, so +angle tilts the streak up-right)."""
    length = max(float(length), 1.0)
    size = size or _odd(length + 3)
    r = (size - 1) / 2
    t = np.deg2rad(angle)
    n = size * oversample
    psf_hi = np.zeros((n, n))
    m = int(np.ceil(length * oversample * 4)) + 1
    s = np.linspace(-length / 2, length / 2, m)
    xs = (s * np.cos(t) + r + 0.5) * oversample - 0.5
    ys = (-s * np.sin(t) + r + 0.5) * oversample - 0.5
    xi = np.clip(np.round(xs).astype(int), 0, n - 1)
    yi = np.clip(np.round(ys).astype(int), 0, n - 1)
    np.add.at(psf_hi, (yi, xi), 1.0)
    psf_hi = ndimage.gaussian_filter(psf_hi, oversample * 0.35)
    psf = psf_hi.reshape(size, oversample, size, oversample).sum(axis=(1, 3))
    return normalize_psf(psf)


def make_psf(kind: str, **p) -> np.ndarray:
    kind = kind.lower()
    if kind == "gaussian":
        return gaussian_psf(p["sigma"], p.get("size"))
    if kind in ("disk", "defocus"):
        return disk_psf(p["radius"], p.get("size"))
    if kind == "motion":
        return motion_psf(p["length"], p.get("angle", 0.0), p.get("size"))
    raise ValueError(f"unknown PSF kind {kind!r}")


# ==========================================================================
# helpers
# ==========================================================================

def center_psf(psf: np.ndarray) -> np.ndarray:
    """Shift a PSF so its centroid sits on the central pixel."""
    psf = normalize_psf(psf)
    cy, cx = ndimage.center_of_mass(psf)
    ty, tx = (psf.shape[0] - 1) / 2, (psf.shape[1] - 1) / 2
    return normalize_psf(ndimage.shift(psf, (ty - cy, tx - cx), order=1, mode="constant"))


def _window(shape):
    return np.outer(np.hanning(shape[0]), np.hanning(shape[1]))


def _log_spectrum(img: np.ndarray) -> np.ndarray:
    """Centered log-magnitude spectrum of a Hann-windowed, square crop."""
    g = to_gray(img).astype(float)
    n = min(g.shape)
    n -= n % 2
    y0, x0 = (g.shape[0] - n) // 2, (g.shape[1] - n) // 2
    g = g[y0:y0 + n, x0:x0 + n]
    g = (g - g.mean()) * _window(g.shape)
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(g))) + 1e-8)


def _radial_profile(a: np.ndarray, nbins: int | None = None):
    n = a.shape[0]
    c = n // 2
    y, x = np.indices(a.shape)
    r = np.hypot(y - c, x - c)
    rb = r.astype(int)
    nbins = nbins or c
    sums = np.bincount(rb.ravel(), a.ravel(), minlength=nbins)[:nbins]
    cnt = np.bincount(rb.ravel(), minlength=nbins)[:nbins]
    prof = sums / np.maximum(cnt, 1)
    f = np.arange(nbins) / n     # cycles / pixel
    return f, prof


# ==========================================================================
# 1. From point sources (stars, beads, pin-holes, specular glints)
# ==========================================================================

def find_point_sources(img: np.ndarray, size: int = 31, n: int = 10,
                       saturation: float = 0.98):
    """Return (row, col) of the brightest isolated, unsaturated local maxima."""
    g = to_gray(img)
    bg = ndimage.median_filter(g, size=size) if min(g.shape) > 3 * size else np.median(g)
    d = g - bg
    peaks = (d == ndimage.maximum_filter(d, size=size)) & (d > 6 * np.std(d))
    h = size // 2
    cand = []
    for r, c in zip(*np.nonzero(peaks)):
        if r < h or c < h or r >= g.shape[0] - h or c >= g.shape[1] - h:
            continue
        if g[r, c] >= saturation:
            continue
        cand.append((d[r, c], r, c))
    cand.sort(reverse=True)
    # isolation: drop points with a neighbour brighter than 20% within the box
    out = []
    for v, r, c in cand:
        if all(abs(r - r2) > size or abs(c - c2) > size for r2, c2 in out):
            out.append((r, c))
        if len(out) >= n:
            break
    return out


def estimate_from_point_source(img: np.ndarray, size: int = 31,
                               centers=None, n: int = 10) -> np.ndarray:
    """Estimate the PSF from images of (unresolved) point sources.

    Each point is cropped, background-subtracted (median of the box rim),
    shifted to sub-pixel centre by its centroid, normalised, and the stack is
    combined by the median (robust to a neighbour or cosmic ray)."""
    g = to_gray(img)
    size = _odd(size)
    h = size // 2
    if centers is None:
        centers = find_point_sources(g, size, n)
    if not centers:
        raise RuntimeError("no isolated point source found")
    stack = []
    for r, c in centers:
        r, c = int(r), int(c)
        crop = g[r - h:r + h + 1, c - h:c + h + 1].astype(float)
        if crop.shape != (size, size):
            continue
        rim = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
        crop = np.clip(crop - np.median(rim), 0, None)
        # suppress residual background with a soft circular mask
        yy, xx = np.mgrid[-h:h + 1, -h:h + 1]
        crop *= np.clip((h + 0.5 - np.hypot(yy, xx)) / 2, 0, 1)
        if crop.sum() <= 0:
            continue
        stack.append(center_psf(crop))
    if not stack:
        raise RuntimeError("point-source crops were empty")
    return center_psf(np.median(np.stack(stack), axis=0))


# ==========================================================================
# 2. From a straight edge (slanted-edge method, ISO 12233 idea)
# ==========================================================================

def _best_edge_roi(g: np.ndarray, win: int = 64):
    """Find the window with the strongest, most coherent (single
    straight edge) gradient structure."""
    gy, gx = np.gradient(ndimage.gaussian_filter(g, 1.0))
    jxx = ndimage.uniform_filter(gx * gx, win)
    jyy = ndimage.uniform_filter(gy * gy, win)
    jxy = ndimage.uniform_filter(gx * gy, win)
    tr = jxx + jyy
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (tr + 1e-12)
    score = tr * coh ** 4
    h = win // 2
    score[:h] = score[-h:] = 0
    score[:, :h] = score[:, -h:] = 0
    r, c = np.unravel_index(np.argmax(score), score.shape)
    return (r - h, c - h, win, win)


def estimate_from_edge(img: np.ndarray, roi=None, return_lsf: bool = False):
    """Estimate an isotropic Gaussian PSF from a sharp straight edge.

    roi : (row, col, height, width) region containing one straight edge.
          Found automatically if None.
    Returns (psf, sigma) or (psf, sigma, (dist, lsf)) when return_lsf."""
    g = to_gray(img).astype(float)
    if roi is None:
        roi = _best_edge_roi(g, win=min(64, min(g.shape) // 2))
    r0, c0, hh, ww = roi
    p = g[r0:r0 + hh, c0:c0 + ww]
    gy, gx = np.gradient(ndimage.gaussian_filter(p, 0.7))
    w = gx ** 2 + gy ** 2
    # edge normal = principal eigenvector of the structure tensor
    J = np.array([[np.sum(gx * gx), np.sum(gx * gy)],
                  [np.sum(gx * gy), np.sum(gy * gy)]])
    evals, evecs = np.linalg.eigh(J)
    nx, ny = evecs[:, 1]
    yy, xx = np.indices(p.shape)
    mag = np.sqrt(w)
    cy, cx = np.sum(mag * yy) / mag.sum(), np.sum(mag * xx) / mag.sum()
    d = (xx - cx) * nx + (yy - cy) * ny
    # 4x oversampled edge spread function
    binw = 0.25
    keep = np.abs(d) < min(hh, ww) / 2 - 1
    bins = np.round(d[keep] / binw).astype(int)
    off = bins.min()
    esf = np.bincount(bins - off, p[keep]) / np.maximum(np.bincount(bins - off), 1)
    cnt = np.bincount(bins - off)
    dist = (np.arange(len(esf)) + off) * binw
    good = cnt > 0
    esf = np.interp(dist, dist[good], esf[good])
    # Fit the ESF directly with an error-function edge.  (The ESF of an
    # isotropic Gaussian PSF is an erf with the same sigma.)  Fitting the ESF
    # avoids differentiating noisy data.
    from scipy.special import erf

    def model(q, x):
        lo, hi, mu, s = q
        return lo + (hi - lo) * 0.5 * (1 + erf((x - mu) / (np.sqrt(2) * s)))
    q0 = [esf[:5].mean(), esf[-5:].mean(), 0.0, 1.5]
    res = optimize.least_squares(lambda q: model(q, dist) - esf, q0,
                                 bounds=([-np.inf, -np.inf, dist.min(), 0.2],
                                         [np.inf, np.inf, dist.max(), 50]))
    lo, hi, mu, s_fit = res.x
    # remove the small known blur added by the 0.25 px bins
    sigma = float(np.sqrt(max(s_fit ** 2 - (0.25 ** 2) / 12, 0.04)))
    # line spread function (for display): derivative of the lightly smoothed ESF
    lsf = np.gradient(ndimage.gaussian_filter1d(esf, 2)) * np.sign(hi - lo)
    dist = dist - mu
    psf = gaussian_psf(sigma)
    if return_lsf:
        return psf, sigma, (dist, lsf)
    return psf, sigma


# ==========================================================================
# 3. Blind parametric estimation from the power spectrum
# ==========================================================================

def estimate_gaussian_blind(img: np.ndarray):
    """Blind Gaussian sigma from the radially averaged power spectrum.

    Model (natural-image statistics):
        P(f) = A f^-alpha * exp(-4 pi^2 sigma^2 f^2) + N
    i.e. a 1/f^alpha scene spectrum attenuated by the Gaussian OTF^2 plus a
    white noise floor.  Fit in log domain.  Returns (psf, sigma)."""
    ls = _log_spectrum(img)
    f, prof = _radial_profile(2 * ls)        # log power
    sel = (f > 0.01) & (f < 0.5)
    f, lp = f[sel], prof[sel]

    def model(q):
        logA, alpha, sigma, logN = q
        return np.logaddexp(logA - alpha * np.log(f) - 4 * np.pi ** 2 * sigma ** 2 * f ** 2, logN)

    best = None
    for s0 in (0.7, 1.5, 3.0, 6.0):
        q0 = [lp[0] + 2 * np.log(f[0]), 2.0, s0, lp[-1]]
        r = optimize.least_squares(lambda q: model(q) - lp, q0,
                                   bounds=([-np.inf, 1.0, 0.0, -np.inf], [np.inf, 3.5, 30, np.inf]))
        if best is None or r.cost < best.cost:
            best = r
    sigma = float(best.x[2])
    return gaussian_psf(max(sigma, 0.3)), sigma


def _whitened_radial(img):
    """Radial log spectrum with the smooth 1/f trend removed - spectral zeros
    of the OTF show up as dips."""
    ls = _log_spectrum(img)
    f, prof = _radial_profile(ls)
    sel = (f > 0.005) & (f < 0.45)
    f, prof = f[sel], prof[sel]
    trend = ndimage.uniform_filter1d(prof, max(5, len(prof) // 6), mode="nearest")
    return f, prof - trend


def estimate_defocus_blind(img: np.ndarray, r_range=(1.0, 25.0),
                           return_score: bool = False):
    """Blind defocus radius: correlate the whitened radial log spectrum with
    the log |OTF| of disks of candidate radius (zeros of 2 J1(x)/x).
    Returns (psf, radius)."""
    from scipy.special import j1
    f, wp = _whitened_radial(img)
    wp = (wp - wp.mean()) / (wp.std() + 1e-12)
    best_r, best_s = r_range[0], -np.inf
    for R in np.arange(r_range[0], r_range[1] + 1e-9, 0.1):
        x = 2 * np.pi * R * f
        otf = np.abs(2 * j1(x) / np.maximum(x, 1e-9))
        m = np.log(otf + 0.02)
        m = m - ndimage.uniform_filter1d(m, max(5, len(m) // 6), mode="nearest")
        m = (m - m.mean()) / (m.std() + 1e-12)
        s = np.mean(m * wp)
        if s > best_s:
            best_s, best_r = s, R
    if return_score:
        return disk_psf(best_r), float(best_r), float(best_s)
    return disk_psf(best_r), float(best_r)


def estimate_motion_blind(img: np.ndarray, max_length: int = 60):
    """Blind linear-motion estimate.

    angle  : motion blur multiplies the spectrum by a sinc along the motion
             direction, producing parallel dark stripes perpendicular to it.
             We project the trend-removed log spectrum at every angle; the
             projection along the stripes oscillates most (Radon method).
    length : the sinc zeros are spaced N/L bins apart, so the cepstrum
             (IFFT of log|G|) has a strong negative peak at distance L along
             the motion direction.
    Returns (psf, length, angle_deg)."""
    ls = _log_spectrum(img)
    n = ls.shape[0]
    c = n // 2
    # remove radial trend
    y, x = np.indices(ls.shape)
    r = np.hypot(y - c, x - c)
    f, prof = _radial_profile(ls)
    trend = np.interp(r, np.arange(len(prof)), ndimage.uniform_filter1d(prof, 5))
    wl = (ls - trend) * (r < 0.45 * n) * (r > 2)
    # Radon-style projection variance
    def proj_var(a):
        rot = ndimage.rotate(wl, -a, reshape=False, order=1)
        return np.var(rot.sum(axis=0)[c - n // 3:c + n // 3])
    coarse = np.arange(0, 180, 2.0)                     # coarse search ...
    a0 = coarse[int(np.argmax([proj_var(a) for a in coarse]))]
    fine = np.arange(a0 - 2, a0 + 2.01, 0.25)            # ... then refine
    best_a = float(fine[int(np.argmax([proj_var(a) for a in fine]))]) % 180
    angle = best_a          # direction (in spectrum) across which values oscillate
    # cepstrum along the motion direction
    g = to_gray(img).astype(float)
    m = min(g.shape); m -= m % 2
    g = g[:m, :m]
    G = np.fft.fft2((g - g.mean()) * _window(g.shape))
    cep = np.real(np.fft.ifft2(np.log(np.abs(G) + 1e-8)))
    cep = np.fft.fftshift(cep)
    cc = m // 2
    t = np.deg2rad(angle)
    rs = np.arange(2, min(max_length, cc - 1), 0.25)
    coords = np.array([cc - rs * np.sin(t), cc + rs * np.cos(t)])
    line = ndimage.map_coordinates(cep, coords, order=1)
    coords2 = np.array([cc + rs * np.sin(t), cc - rs * np.cos(t)])
    line = 0.5 * (line + ndimage.map_coordinates(cep, coords2, order=1))
    # the blur shows up as the most negative cepstral value (first sinc-zero
    # period); restrict to r >= 3 where the central lobe has decayed
    sel = rs >= 3
    length = float(rs[sel][np.argmin(line[sel])])
    # PSF angle convention: CCW from +x on screen.  The spectrum stripes are
    # perpendicular to motion, and the oscillation direction == motion dir.
    psf_angle = angle % 180
    return motion_psf(length, psf_angle), length, float(psf_angle)


# ==========================================================================
# 4. Non-parametric blind kernel estimation
# ==========================================================================

def _l0_latent(y, k, lam, beta_max=1e5):
    """L0-gradient latent image (Xu et al. 2013 / Pan et al. 2014):

        argmin_x ||k*x - y||^2 + lam ||grad x||_0

    solved by half-quadratic splitting.  The result keeps only salient
    edges and flattens texture/noise - exactly what kernel estimation needs."""
    K = psf2otf(k, y.shape)
    Dx = psf2otf(np.array([[1.0, -1.0]]), y.shape)
    Dy = psf2otf(np.array([[1.0], [-1.0]]), y.shape)
    KtY = np.conj(K) * _sfft.fft2(y)
    K2 = np.abs(K) ** 2
    D2 = np.abs(Dx) ** 2 + np.abs(Dy) ** 2
    x = y.copy()
    beta = 2 * lam
    while beta < beta_max:
        h = np.roll(x, -1, 1) - x
        v = np.roll(x, -1, 0) - x
        m = (h ** 2 + v ** 2) < lam / beta
        h[m] = 0
        v[m] = 0
        rhs = KtY + beta * (np.conj(Dx) * _sfft.fft2(h, workers=-1) + np.conj(Dy) * _sfft.fft2(v, workers=-1))
        x = np.real(_sfft.ifft2(rhs / (K2 + beta * D2), workers=-1))
        beta *= 2
    return x


def _kernel_step(x, y, ksize, gamma=2.0):
    """Least squares kernel in the gradient domain:

        argmin_k ||grad(x) * k - grad(y)||^2 + gamma ||k||^2

    followed by projection onto PSF constraints (>= 0, sparse, connected,
    sum 1, centred)."""
    # gradients, windowed so the FFT's periodic wrap-around does not create
    # false edges (this matters a lot for kernel accuracy)
    w = np.sqrt(_window(y.shape))
    g = lambda a: ((np.roll(a, -1, 1) - a) * w, (np.roll(a, -1, 0) - a) * w)
    xs, ys = g(x), g(y)
    num = sum(np.conj(_sfft.fft2(a)) * _sfft.fft2(b) for a, b in zip(xs, ys))
    den = sum(np.abs(_sfft.fft2(a)) ** 2 for a in xs) + gamma
    k = np.fft.fftshift(np.real(_sfft.ifft2(num / den)))
    cy, cx = y.shape[0] // 2, y.shape[1] // 2
    h = ksize // 2
    k = k[cy - h:cy + h + 1, cx - h:cx + h + 1]
    k = np.clip(k, 0, None)
    if k.max() <= 0:
        return None
    k[k < 0.05 * k.max()] = 0
    lab, nl = ndimage.label(k > 0, structure=np.ones((3, 3)))
    if nl > 1:   # drop small isolated specks
        mass = ndimage.sum(k, lab, range(1, nl + 1))
        for i, mm in enumerate(mass):
            if mm < 0.1 * mass.max():
                k[lab == i + 1] = 0
    return center_psf(k)


def blind_estimate(img: np.ndarray, size: int = 25, lam: float = 4e-3,
                   iters_per_level: int = 5, gamma: float = 2.0,
                   max_dim: int = 512, verbose: bool = False) -> np.ndarray:
    """Estimate an arbitrary-shaped blur kernel from one blurred image.

    Coarse-to-fine alternating minimisation (Fergus 2006, Cho & Lee 2009,
    Pan et al. 2014):
      1. latent image  x <- argmin ||k*x - y||^2 + lam ||grad x||_0
      2. kernel        k <- argmin ||grad x * k - grad y||^2 + gamma ||k||^2
      3. project k: non-negative, sparse, connected, unit sum, centred
    The image pyramid (factor sqrt 2) lets a large blur be found first as a
    small one.  `size` must exceed the expected blur extent.  Images larger
    than `max_dim` are centre-cropped for speed (the blur is assumed to be
    the same everywhere)."""
    y0 = to_gray(img).astype(float)
    if max(y0.shape) > max_dim:
        r0 = max(0, (y0.shape[0] - max_dim) // 2)
        c0 = max(0, (y0.shape[1] - max_dim) // 2)
        y0 = y0[r0:r0 + max_dim, c0:c0 + max_dim]
    size = _odd(size)
    # pyramid sizes: kernel shrinks by sqrt(2) until ~3-5 px
    ksizes = [size]
    while ksizes[-1] > 5:
        ksizes.append(_odd(ksizes[-1] / np.sqrt(2)))
    ksizes = ksizes[::-1]
    k = None
    for li, ks in enumerate(ksizes):
        scale = ks / size
        ys = ndimage.zoom(y0, scale, order=1) if scale < 1 else y0
        if k is None:
            k = np.zeros((ks, ks)); k[ks // 2, ks // 2 - 1:ks // 2 + 2] = 1
            k[ks // 2 - 1:ks // 2 + 2, ks // 2] = 1
            k = normalize_psf(k)
        else:
            k = ndimage.zoom(k, ks / k.shape[0], order=1)
            if k.shape[0] != ks:   # rounding safety
                k = np.pad(k, ((0, max(0, ks - k.shape[0])),) * 2)[:ks, :ks]
            k = center_psf(np.clip(k, 0, None))
        H = _sfft.next_fast_len(ys.shape[0] + 2 * ks)
        W = _sfft.next_fast_len(ys.shape[1] + 2 * ks)
        yp = np.pad(ys, ((ks, H - ys.shape[0] - ks), (ks, W - ys.shape[1] - ks)), mode="reflect")
        l = lam
        for it in range(iters_per_level):
            x = _l0_latent(yp, k, l)
            knew = _kernel_step(x, yp, ks, gamma)
            if knew is not None:
                k = knew
            l = max(l / 1.1, 1e-4)
        if verbose:
            print(f"level {li + 1}/{len(ksizes)}  kernel {ks}x{ks}")
    return normalize_psf(k)


# ==========================================================================
# Dispatcher
# ==========================================================================

def estimate_psf(img: np.ndarray, kind: str = "auto", **kw):
    """Estimate a PSF.

    kind : 'point'    from point sources            (kw: size, centers)
           'edge'     from a straight edge          (kw: roi)
           'gaussian' blind Gaussian (spectrum)
           'defocus'  blind disk radius (spectrum zeros)
           'motion'   blind linear motion (Radon + cepstrum)
           'blind'    non-parametric kernel         (kw: size)
           'auto'     run the parametric blind estimators and keep the one
                      whose model best explains the spectrum
    Returns (psf, info_dict)."""
    kind = kind.lower()
    if kind == "point":
        return estimate_from_point_source(img, **kw), {"kind": "point"}
    if kind == "edge":
        psf, s = estimate_from_edge(img, **kw)
        return psf, {"kind": "gaussian", "sigma": s, "source": "edge"}
    if kind == "gaussian":
        psf, s = estimate_gaussian_blind(img)
        return psf, {"kind": "gaussian", "sigma": s}
    if kind in ("defocus", "disk"):
        psf, r = estimate_defocus_blind(img, **kw)
        return psf, {"kind": "disk", "radius": r}
    if kind == "motion":
        psf, L, a = estimate_motion_blind(img, **kw)
        return psf, {"kind": "motion", "length": L, "angle": a}
    if kind == "blind":
        return blind_estimate(img, **kw), {"kind": "blind"}
    if kind == "auto":
        return _auto(img)
    raise ValueError(f"unknown kind {kind!r}")


def _spectrum_fit_score(img, psf):
    """How well log|OTF| explains the whitened 2-D log spectrum
    (correlation, higher is better)."""
    ls = _log_spectrum(img)
    n = ls.shape[0]
    otf = np.fft.fftshift(np.abs(psf2otf(psf, (n, n))))
    m = np.log(otf + 0.02)
    c = n // 2
    y, x = np.indices(ls.shape)
    r = np.hypot(y - c, x - c)
    sel = (r > 3) & (r < 0.45 * n)
    # remove radial trend from the image spectrum
    f, prof = _radial_profile(ls)
    t = np.interp(r, np.arange(len(prof)), ndimage.uniform_filter1d(prof, 9))
    a = (ls - t)[sel]
    f2, prof2 = _radial_profile(m)
    t2 = np.interp(r, np.arange(len(prof2)), ndimage.uniform_filter1d(prof2, 9))
    b = (m - t2)[sel]
    if b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _auto(img, motion_thr: float = 0.28, disk_thr: float = 0.55):
    """Pick the PSF family automatically.

    * motion : 2-D correlation between the whitened log spectrum and the
               estimated motion OTF (straight parallel zero-stripes).
    * disk   : 1-D correlation of the radial spectrum with Bessel zero rings.
    * Gaussian OTFs have no zeros, so when neither zero pattern is found the
      blind Gaussian fit is used.
    Thresholds were calibrated on synthetic blurs of natural images."""
    pm, L, a = estimate_motion_blind(img)
    ms = _spectrum_fit_score(img, pm)
    pd, R, ds = estimate_defocus_blind(img, return_score=True)
    scores = {"motion": round(ms, 3), "disk": round(ds, 3)}
    if ms >= motion_thr:
        return pm, {"kind": "motion", "length": L, "angle": a, "scores": scores}
    if ds >= disk_thr:
        return pd, {"kind": "disk", "radius": R, "scores": scores}
    pg, sg = estimate_gaussian_blind(img)
    return pg, {"kind": "gaussian", "sigma": sg, "scores": scores}
