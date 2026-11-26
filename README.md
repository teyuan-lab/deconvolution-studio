# Deconvolution Studio

**Live site: https://teyuan-lab.github.io/deconvolution-studio/**

Deconvolution Studio estimates the point spread function (PSF) of a blurred image and removes the blur. Everything runs in your browser: images are never uploaded, and nothing needs to be installed.

## Using the site

1. **Image.** Open a photo, drop or paste it onto the viewer, or start with a sample.
2. **Test blur** (optional). Blur a sharp image yourself with Gaussian, defocus or motion blur plus noise. Quality scores (PSNR / SSIM) against the original are then shown.
3. **Point spread function.** Choose where the PSF comes from:
   - auto estimate
   - Gaussian, defocus or motion, entered by hand
   - blind kernel, for any shape such as camera shake
   - point sources such as stars or beads
   - a straight edge
   - a PSF file (image or CSV)
4. **Deblur.** Choose an algorithm: total variation, Richardson–Lucy, Richardson–Lucy + TV, Tikhonov or Wiener. Adjust the regularisation if you want, then download the result.

**Viewer controls**

| Action | Control |
|---|---|
| Zoom | scroll, or pinch on a touch screen |
| Pan | drag |
| Fit to window | double-click, or key 0 |
| Compare before and after | Split view: drag the red divider |
| Switch views | keys 1–4 |

The default working size is 512 px so that results come back in a few seconds. Choose a larger size or full resolution in step 1 for the final result. Colour channels are processed in parallel Web Workers.

## How it is built

The site is plain static HTML, CSS and JavaScript modules. It has no build step and no dependencies, and GitHub Pages serves the repository root as-is.

| Path | Contents |
|---|---|
| `index.html`, `css/`, `js/app.js` | interface, viewer, worker orchestration |
| `js/worker.js` | runs all computation off the main thread |
| `js/lib/fft.js` | mixed-radix FFT (radix 2/3/4/5 + generic) |
| `js/lib/core.js` | image helpers matching numpy / scipy.ndimage behaviour |
| `js/lib/psf.js` | PSF models and the estimators: auto (spectrum), blind L0 kernel, point source, slanted edge |
| `js/lib/deconvolve.js` | Wiener, Tikhonov (GCV), Richardson–Lucy, RL-TV, TV-ADMM with unknown-boundary handling |
| `python/` | the original Python toolkit, desktop app (`app.py`), Windows `.exe` build script, demos and tests |
| `ALGORITHMS.md` | study notes: theory, algorithm choices, measured results |

The JavaScript library is a direct port of the Python toolkit in `python/deconv/`. I checked it against the Python version on identical inputs. The PSF models agree to 1e-16. The blind-kernel result is identical. PSF estimates and PSNR agree to the reported precision.

## Python version

`python/` has the same algorithms as a Python package, plus the Tkinter desktop app and a script that builds a stand-alone Windows `.exe`. See [`python/README.md`](python/README.md) and [`python/INSTALL.txt`](python/INSTALL.txt).

```
cd python
pip install -r requirements.txt
python app.py
```
