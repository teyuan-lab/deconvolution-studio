# Deconvolution Toolkit

This toolkit estimates the point spread function (PSF) of a blurred image and deconvolves the image with it. It is written in pure Python and needs only **numpy, scipy and matplotlib**.

```
pip install -r requirements.txt
python app.py           # Deconvolution Studio - the main desktop interface
python demo.py          # batch demo -> demo_output/*.png + results.json  (~1-2 min)
python gui_demo.py      # minimal matplotlib demo window
python tests/test_deconv.py
```

On Windows you can also double-click **`Deconvolution Studio.bat`**.

### Stand-alone .exe (no Python on the target PC)

Double-click **`build_exe.bat`** once on a PC that has Python. It does four things:

1. Creates a clean virtual environment.
2. Installs numpy, scipy, Pillow and PyInstaller.
3. Builds the app from `DeconvolutionStudio.spec`.
4. Runs `DeconvolutionStudio.exe --selftest` to check the build.

The result is `DeconvolutionStudio.exe`, a single file of about 60 MB, plus `DeconvolutionStudio-Windows.zip`, which contains the same exe zipped for e-mailing. Both run on any 64-bit Windows 10/11 PC without installing Python.

The exe is not code-signed, so SmartScreen may ask you to confirm: click *More info*, then *Run anyway*.

## Deconvolution Studio (`app.py`)

The app is laid out as four numbered steps in the left panel. Each step is
optional except opening an image.

1. **Image.** Open a file, or pick one of the bundled test images. *Working size* downsizes large photos so you can experiment quickly. Switch it to *Full resolution* for the final result.
2. **Test blur.** Blur a sharp image with a Gaussian, defocus or motion PSF plus noise. While a test blur is active, quality scores (PSNR/SSIM) against the original are shown.
3. **Point spread function.** Choose where the PSF comes from: auto estimate, Gaussian/defocus/motion entered by hand, blind kernel, point sources, straight edge, or a PSF file. A preview of the PSF and its parameters is shown.
4. **Deblur.** Choose the algorithm, the regularisation multiplier (log scale) and the number of iterations, then press **Deblur** (F5). If no PSF exists yet, it is estimated first. Heavy work runs in the background, so the window stays responsive.

**Viewer controls**

| Action | Control |
|---|---|
| Change view | Input / Result / **Split** (drag the divider) / Side by side, or keys 1-4 |
| Zoom | mouse wheel |
| Pan | left-drag |
| Fit to window | double-click, or key 0 |
| Show at 100% | 100% button |

**Other actions**

| Action | Control |
|---|---|
| Save the result | Ctrl+S |
| Save the PSF | File > Save PSF (.npy / .csv / .png) |
| Refine iteratively | "Use result as input" |
| Read the notes | Help > Algorithm notes |

The theory, the algorithm choices and the measured results are written up in **[ALGORITHMS.md](ALGORITHMS.md)**.

## Layout

| Path | Contents |
|---|---|
| `deconv/psf.py` | PSF models; point-source, edge, blind Gaussian/defocus/motion and blind-kernel estimators; `estimate_psf(..., "auto")` |
| `deconv/deconvolve.py` | Wiener, Tikhonov (GCV), Richardson–Lucy, RL-TV, TV-ADMM with unknown-boundary handling |
| `deconv/utils.py` | FFT helpers, edge taper, noise estimate, image I/O, PSNR/SSIM |
| `deconv/cli.py` | command line (`python -m deconv ...`) |
| `demo.py` | reproducible demo figures |
| `app.py` | **Deconvolution Studio** desktop interface (Tkinter) |
| `gui_demo.py` | minimal interactive matplotlib demo |
| `cameraman.png`, `astronaut.png` | test images (scikit-image sample data, public domain) |

## Command line

```bash
# 1) Estimate a PSF  (auto | gaussian | defocus | motion | blind | point | edge)
python -m deconv estimate photo.jpg --method auto -o psf.npy
python -m deconv estimate shake.jpg --method blind --size 35 -o psf.png
python -m deconv estimate stars.tif --method point --size 21 -o psf.npy

# 2) Deconvolve with a known PSF, or estimate and deblur in one step
python -m deconv deconvolve photo.jpg --psf psf.npy --method tv -o sharp.png
python -m deconv deconvolve photo.jpg --gaussian 2.0 --method rl --iterations 40 -o sharp.png
python -m deconv deconvolve photo.jpg --estimate auto --method tv --save-psf psf.png -o sharp.png
python -m deconv deconvolve photo.jpg --estimate blind --size 31 --method tv --lam 1e-3 -o sharp.png

# 3) Make test data
python -m deconv simulate sharp.png --motion 15 30 --noise 0.01 -o blurred.png
```

Available `--method` choices for deconvolution: `wiener`, `tikhonov`, `rl`, `rl_tv`, `tv` (the default), and `inverse` (didactic only).

## Python API

```python
import deconv as D

y = D.load_image("photo.jpg")                         # float in [0,1], gray or RGB
psf, info = D.estimate_psf(y, "auto")                 # or "motion", "blind", "point", ...
print(info)                                           # {'kind': 'motion', 'length': 17.25, 'angle': 35.0, ...}
x = D.deconvolve(y, psf, "tv")                        # lam chosen automatically (GCV)
x = D.deconvolve(y, psf, "tv", lam=2e-3)              # or set it by hand
D.save_image("sharp.png", x)

# PSF models
D.gaussian_psf(2.0); D.disk_psf(5); D.motion_psf(15, 30)
```

## Which estimator should I use?

| You have… | Use |
|---|---|
| stars, beads, pinholes or other isolated points | `point` (most accurate; handles any PSF shape) |
| a straight, sharp, high-contrast edge | `edge` (Gaussian σ to ±0.02 px) |
| a general photo with defocus, motion or soft blur | `auto` (chooses between Gaussian, disk and motion) |
| camera shake with a curved or irregular path | `blind` (set `--size` larger than the blur) |

## Tips

- Run the tool on the original image, not on one that has already been sharpened or downscaled.
- If the result rings, increase `--lam`, or in the GUI move the "regularisation ×" slider to the right.
- If the result looks flat or cartoon-like, decrease the regularisation.
- Images larger than 1024 px are downsized in the GUI only. The command line processes full resolution.
