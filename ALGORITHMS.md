# Image Deconvolution: Study Notes

These notes cover the theory behind the `deconv` toolkit in this folder. Each section names the function that implements the method, and the numbers in the last section come from `demo.py`.

---

## 1. The forward model

A blurred image is modelled as

```
y = h * x + n
```

| symbol | meaning |
|---|---|
| `x` | the unknown sharp image |
| `h` | the **point spread function (PSF)**, i.e. the image of a single point of light |
| `*` | 2-D convolution |
| `n` | noise: Gaussian (read noise) and/or Poisson (photon shot noise) |

In the Fourier domain convolution becomes multiplication:

```
Y(f) = H(f) · X(f) + N(f)
```

`H = FFT(h)` is the **optical transfer function (OTF)**. Its magnitude, the MTF, says how much contrast survives at each spatial frequency.

**Assumptions.** The model assumes the blur is the same everywhere in the image (spatially invariant) and that the image is linear in light. Gamma-encoded JPEGs violate the second assumption mildly. Saturated pixels and blur that varies across the frame violate the model more seriously.

### Why deconvolution is hard (ill-posedness)

The naive solution `X = Y / H` fails for three reasons:

1. **`H` is small at high frequencies.** At those frequencies `N/H` is enormous, so the noise gets amplified without bound.
2. **`H` has exact zeros for some blurs.** A defocus disk has rings of zeros, and linear motion has parallel stripes of zeros. Information at those frequencies is gone.
3. **Boundaries are unknown.** The FFT assumes the image is periodic. The mismatch between the left and right (or top and bottom) edges creates ringing that spreads across the frame.

Every practical method therefore adds **prior knowledge** (regularisation) about what sharp images look like, and handles the image boundary explicitly.

---

## 2. PSF models (`deconv/psf.py`)

| Model | Physical cause | Shape | OTF |
|---|---|---|---|
| **Gaussian** `gaussian_psf(sigma)` | lens aberrations, atmosphere, sensor/AA filter | smooth bell | Gaussian; no zeros |
| **Disk** `disk_psf(R)` | defocus (geometric optics) | uniform pillbox of radius R | `2·J1(2πRρ)/(2πRρ)`; zero rings at ρ ≈ 0.61/R, 1.12/R, … |
| **Linear motion** `motion_psf(L, θ)` | camera or object moving during exposure | line segment of length L at angle θ | `sinc(L·f·u)` along the motion direction u; zero stripes spaced 1/L apart |
| **Arbitrary kernel** | camera shake, complex optics | any non-negative shape with unit sum | anything |

Every PSF is **non-negative** and **sums to 1**, so blurring preserves total energy. All estimators enforce both properties.

---

## 3. Non-blind deconvolution (`deconv/deconvolve.py`)

"Non-blind" means the PSF is known, either measured or estimated first.

### 3.1 Inverse filter: `inverse_filter`
`X = Y / H`, set to 0 wherever `|H| < threshold`. It is included only to demonstrate noise amplification.

### 3.2 Wiener filter: `wiener`
This is the linear minimum-mean-square-error (MMSE) estimator when signal and noise are stationary:

```
X = conj(H) · Y / (|H|² + NSR)
```

NSR is the noise-to-signal power ratio. In this toolkit it is a scalar set to `60·σ²`, where σ is a robust estimate of the noise standard deviation (described in §6). The method is closed-form and fast, but a scalar NSR treats all frequencies alike. The result therefore tends to keep some noise "texture" and to ring near edges.

### 3.3 Tikhonov / constrained least squares: `tikhonov`

```
x̂ = argmin ||h*x − y||² + λ ||∇x||²
X = conj(H) · Y / (|H|² + λ(|Dx|² + |Dy|²))
```

The gradient penalty grows like `f²`, which matches the `1/f²` power spectrum of natural images. It suppresses high-frequency noise more strongly than low-frequency noise, so it beats Wiener with a flat NSR. This is the frequency-domain form of MATLAB's `deconvreg`.

### 3.4 Richardson–Lucy (RL): `richardson_lucy`
RL is the maximum-likelihood estimate under **Poisson** noise, written as a multiplicative EM iteration:

```
x_{k+1} = x_k · Hᵀ( y / (H x_k) )
```

- It keeps the image non-negative and preserves total flux, which is why it is standard in astronomy and fluorescence microscopy.
- It **semi-converges**. PSNR rises, peaks, and then falls as noise is amplified (demo figure 6: peak at about 40 iterations). The iteration count acts as the regulariser.
- **RL-TV** (Dey et al. 2006) divides by `1 − λ·div(∇x/|∇x|)`. This stabilises the iteration so it can run to convergence (figure 6: still rising at 150 iterations).

### 3.5 Total-variation deconvolution (ADMM): `tv_admm`

```
x̂ = argmin ½||M(h*x) − y||² + λ Σ |∇x|
```

TV is an L1 penalty on the gradient. It favours piecewise-smooth images: edges stay sharp, and flat regions are cleaned of noise. The method is solved with **ADMM** using two splits, `v = h*x` and `z = ∇x`:

- **x-step:** `(μ|H|² + ρ|D|²) X = μ Hᵀ(v−u₁) + ρ Dᵀ(z−u₂)`. This is a single FFT division.
- **v-step:** `v = (M·y + μ(Hx+u₁)) / (M + μ)`, computed pixel by pixel.
- **z-step:** isotropic soft-shrinkage of `∇x + u₂` by `λ/ρ`.

The mask `M` gives correct handling of unknown boundaries (§5). On average TV was the best method in this toolkit's tests. Its typical weakness is a "cartoon" look in fine texture.

### 3.6 Other families (not implemented here; useful to know)

- **Hyper-Laplacian priors** (Krishnan & Fergus 2009): an `|∇x|^0.8` penalty, closer to real gradient statistics than TV.
- **Plug-and-play / RED:** replace the prior step of ADMM with any denoiser, such as BM3D or a CNN.
- **Deep learning:** DeblurGAN, MPRNet, Restormer, NAFNet. These give state-of-the-art results but need training data and can hallucinate detail.
- **Spatially varying blur:** process overlapping patches, each with its own PSF, or use an efficient filter-flow model.

---

## 4. Choosing the regularisation weight

| Strategy | Idea | Used here |
|---|---|---|
| Noise-based rule | λ ∝ σ² (Wiener), calibrated on test images | Wiener default |
| **Generalised cross-validation (GCV)** | minimise `‖(1−F)Y‖² / (Σ(1−F))²`, where `F = |H|²/(|H|²+λ|R|²)`; needs no noise estimate | Tikhonov default; TV uses 0.25 × the GCV value |
| Discrepancy principle | choose λ so that `‖h*x − y‖² ≈ N·σ²` | — |
| L-curve | the corner of log residual vs. log penalty | — |

In the tests, the best TV weight was 0.1 to 0.3 times the best Tikhonov weight across images and PSFs. That observation is the basis for the `0.25 × GCV` rule. In the GUI, the "regularisation ×" slider multiplies the automatic value.

---

## 5. Boundary handling

The FFT treats the image as a torus, so the left edge touches the right edge. The toolkit offers two remedies:

1. **Reflect-pad + edge-taper** (`utils.prepare`). The image is padded by the PSF size and its outer border is blended with a blurred copy, making it nearly periodic. Used by Wiener and Tikhonov.
2. **Unknown boundary / masked data term** (`boundary="unknown"`, the default for RL and TV). The image is placed on a larger canvas and only the observed pixels enter the data term, so pixels outside the frame are estimated freely. This removes boundary ringing almost entirely. On the astronaut image with 17 px motion blur, PSNR rose from **24.2 dB to 30.4 dB**.

All FFTs are padded to fast sizes (`scipy.fft.next_fast_len`); this made TV about 3× faster.

---

## 6. Estimating the PSF

Five estimators are implemented. Use the measured ones whenever the scene allows; they are more reliable than the blind ones.

### 6.1 Point source (`estimate_from_point_source`)
This is the gold standard when stars, fluorescent beads, pinholes or specular glints are available. The procedure:

1. Detect isolated, unsaturated local maxima.
2. Crop each one and subtract the background, taken as the median of the crop's rim.
3. Shift each crop to sub-pixel centre using its centroid.
4. Take the **median** of the stack, which tolerates neighbouring objects and cosmic rays.

This method can recover arbitrary and asymmetric PSFs; demo figure 4 shows a comatic one.

### 6.2 Slanted edge (`estimate_from_edge`)
This follows the ISO 12233 idea. A straight, high-contrast edge is the integral of the PSF along one direction. The procedure:

1. Find the edge region automatically, choosing the window with the strongest and most coherent structure tensor.
2. Project pixels onto the edge normal at 0.25 px bins. Because the edge is slanted, this gives super-resolution.
3. Fit an erf to the edge spread function (ESF), which yields σ.

The fit is done on the ESF itself rather than its derivative (the LSF), because differentiating noisy data amplifies the noise. Accuracy is about ±0.02 px (figure 5).

### 6.3 Blind Gaussian (`estimate_gaussian_blind`)
Natural images have a power spectrum of roughly `A·f^−α` with α ≈ 2. The blurred spectrum is fitted with

```
P(f) = A f^−α exp(−4π²σ²f²) + N
```

(scene × OTF² + noise floor), in the log domain, on the radially averaged spectrum.

### 6.4 Blind defocus (`estimate_defocus_blind`)
The radial log spectrum is whitened by subtracting its smooth trend, which leaves dips at the OTF zeros. It is then correlated with `log|2J1(2πRρ)/(2πRρ)|` for candidate radii R.

### 6.5 Blind linear motion (`estimate_motion_blind`)
- **Angle:** the motion OTF draws parallel dark stripes on the spectrum. A Radon-style projection of the whitened log spectrum oscillates most when it runs along the stripes. The search is coarse at 2° steps, then refined at 0.25°.
- **Length:** the zeros are spaced N/L bins apart, so the **cepstrum** `IFFT(log|Y|)` has a strong negative peak at distance L along the motion direction.

### 6.6 Automatic family selection (`estimate_psf(img, "auto")`)
Both zero-pattern models are scored: a 2-D correlation for the motion stripes and a 1-D correlation for the disk rings. If neither explains the spectrum, the blur is assumed to be Gaussian, because a Gaussian OTF has no zeros. The thresholds were calibrated on synthetic blurs; auto-selection was correct in every calibration case.

### 6.7 Non-parametric blind kernel (`blind_estimate`)
This estimates an arbitrary kernel such as camera shake.

**Why naive joint MAP fails.** Minimising over both x and k with a sparse prior favours the trivial solution of a delta kernel and a blurred image (Levin et al., CVPR 2009). Working methods avoid this by estimating k from **salient edges** of an intermediate image that is deliberately over-sharpened.

The algorithm follows Fergus 2006, Cho & Lee 2009, Xu 2013 and Pan 2014:

1. **Pyramid.** Kernel sizes shrink by √2 per level down to about 5 px. A large blur is found first as a small one at the coarse scale, which avoids local minima.
2. **Latent image.** `x = argmin ‖k*x − y‖² + λ‖∇x‖₀`, solved by half-quadratic splitting. The L0 gradient prior keeps only strong edges and flattens texture.
3. **Kernel step.** `k = argmin ‖∇x * k − ∇y‖² + γ‖k‖²`, solved in closed form with an FFT. The gradients are **Hann-windowed** first. Without the window, wrap-around edges corrupt the kernel; ablation showed kernel L1 error fall from 0.47 to 0.04.
4. **Projection.** Clip negatives, zero values below 5 % of the maximum, remove small disconnected specks, normalise, and re-centre.
5. **Final pass.** Run the non-blind TV deconvolution with the estimated kernel.

Limitations: the method needs edges in the image, assumes uniform blur, and is weaker for smooth blurs such as disk and Gaussian. Use the parametric estimators for those.

---

## 7. Results from `demo.py`

These are synthetic tests with known ground truth: 512 × 512 images, PSNR in dB.

**Known Gaussian PSF (σ = 2.5, 1 % noise), cameraman. Blurred: 24.78 dB.**

| Method | PSNR | SSIM | Time |
|---|---|---|---|
| Inverse filter (\|H\| > 0.01) | 19.20 | 0.196 | 0.04 s |
| Wiener | 27.12 | 0.693 | 0.04 s |
| Tikhonov (GCV) | 27.17 | 0.695 | 0.2 s |
| Richardson–Lucy (30 it) | 26.87 | 0.746 | 0.5 s |
| RL-TV (50 it) | 27.11 | 0.750 | 1.1 s |
| **TV-ADMM** | **27.62** | **0.767** | 3.0 s |

**Blind parametric estimation + TV, colour astronaut, 0.5 % noise.**

| True blur | Estimated | Blurred | Deblurred (estimated PSF) | Deblurred (true PSF) |
|---|---|---|---|---|
| Gaussian σ = 2.0 | σ = 2.01 | 24.96 | 29.47 | 29.47 |
| Disk R = 5 | R = 5.1 | 22.87 | 29.45 | 29.48 |
| Motion L = 17, 35° | L = 17.25, 35.0° | 21.00 | 30.02 | 30.31 |

**Other estimators**

| Test | Result |
|---|---|
| Blind kernel, camera-shake trajectory | 21.35 → 28.39 dB (true kernel: 31.26 dB), about 6 s |
| Slanted edge, σ = 1.0 / 2.0 / 3.5 | estimated 1.00 / 1.99 / 3.49 |
| Point source, comatic PSF | L1 error 0.22 from 10 stars |

---

## 8. Practical guidance

- **Measure the PSF if you can.** Use a star or bead, or a photo of a printed slanted edge or point grid taken with the same settings.
- **Work in linear light** if possible: RAW or 16-bit, without sharpening, and with no saturated pixels. If a JPEG is all you have, undo the gamma first by applying `img**2.2`.
- **Start with TV** for photos, and **RL / RL-TV** for photon-limited data such as astronomy and microscopy.
- **Ringing near edges** means the PSF is too large or wrong, or λ is too low. **A plastic or cartoon look** means λ is too high.
- **Spatially varying blur** (rotation, depth-dependent defocus): split the image into tiles, estimate a PSF per tile, and blend the results.

## References

- N. Wiener, *Extrapolation, Interpolation, and Smoothing of Stationary Time Series*, 1949.
- W. H. Richardson, "Bayesian-based iterative method of image restoration," JOSA 1972; L. B. Lucy, AJ 1974.
- L. Rudin, S. Osher, E. Fatemi, "Nonlinear total variation based noise removal," Physica D 1992.
- N. Dey et al., "Richardson–Lucy algorithm with total variation regularization," Microscopy Research & Technique 2006.
- G. Golub, M. Heath, G. Wahba, "Generalized cross-validation as a method for choosing a good ridge parameter," Technometrics 1979.
- S. Boyd et al., "Distributed optimization… ADMM," Foundations & Trends in ML 2011.
- M. Almeida, M. Figueiredo, "Deconvolving images with unknown boundaries using ADMM," IEEE TIP 2013.
- R. Fergus et al., "Removing camera shake from a single photograph," SIGGRAPH 2006.
- A. Levin et al., "Understanding and evaluating blind deconvolution algorithms," CVPR 2009.
- S. Cho, S. Lee, "Fast motion deblurring," SIGGRAPH Asia 2009.
- L. Xu, S. Zheng, J. Jia, "Unnatural L0 sparse representation for natural image deblurring," CVPR 2013.
- J. Pan et al., "Deblurring text images via L0-regularized intensity and gradient prior," CVPR 2014.
- D. Krishnan, R. Fergus, "Fast image deconvolution using hyper-Laplacian priors," NIPS 2009.
- ISO 12233, *Photography – Electronic still picture imaging – Resolution and spatial frequency responses*.
