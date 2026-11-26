"""Deconvolution Studio - desktop interface for the `deconv` toolkit.

    python app.py              # start empty (opens the bundled test image)
    python app.py photo.jpg    # start with your own image

Uses only the standard library's Tkinter plus numpy / scipy / Pillow.

Workflow
--------
1. Open an image (File > Open, Ctrl+O, or the button).
2. (optional) Apply a synthetic test blur to see how well it is undone -
   quality scores against the sharp original are then shown.
3. Pick how to get the PSF (auto estimate, manual model, blind kernel,
   point sources, straight edge, or a PSF file) and press "Get PSF".
4. Pick an algorithm and press "Deblur" (F5).  If no PSF exists yet it is
   estimated first.
5. Inspect with the viewer: Input / Result / Split (drag the divider) /
   Side by side.  Mouse wheel zooms, left-drag pans, double-click fits.
6. Save the result (Ctrl+S) and/or the PSF.
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
import traceback
import webbrowser

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import numpy as np
    import scipy  # noqa: F401
    from PIL import Image, ImageTk
except ImportError as _e:  # pragma: no cover - shown to the user
    _msg = (f"A required package is missing: {_e.name}\n\n"
            "Open a command prompt in this folder and run:\n\n"
            "    python -m pip install -r requirements.txt")
    print(_msg, file=sys.stderr)
    try:
        _r = tk.Tk(); _r.withdraw(); messagebox.showerror("Deconvolution Studio", _msg); _r.destroy()
    except Exception:
        pass
    sys.exit(1)

import deconv as D
from deconv.deconvolve import gcv_lambda
from deconv.utils import edgetaper, estimate_noise_sigma, to_gray

HERE = os.path.dirname(os.path.abspath(__file__))           # bundled data (images, docs)
# writable folder for logs: next to the .exe when frozen by PyInstaller
APP_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else HERE
IMG_TYPES = [("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp"), ("All files", "*.*")]

PSF_SOURCES = ["Auto estimate", "Gaussian (manual)", "Defocus disk (manual)", "Motion (manual)",
               "Blind kernel", "Point sources", "Straight edge", "From file"]
ALGORITHMS = ["TV-ADMM (recommended)", "Richardson-Lucy", "RL-TV", "Tikhonov", "Wiener"]
PREVIEW_SIZES = ["768 px", "1024 px", "1600 px", "Full resolution"]


# ==========================================================================
# helpers
# ==========================================================================

def to_pil(img: np.ndarray) -> Image.Image:
    a = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    return Image.fromarray(a, "L" if a.ndim == 2 else "RGB")


# 'inferno' colour map anchor points (from matplotlib), so the app itself
# does not need matplotlib
_INFERNO = np.array([
    [0, 0, 4], [22, 11, 57], [66, 10, 104], [106, 23, 110], [147, 38, 103],
    [188, 55, 84], [221, 81, 58], [243, 120, 25], [252, 165, 10], [246, 215, 70],
    [252, 255, 164]], dtype=float)


def psf_to_pil(psf: np.ndarray, size: int) -> Image.Image:
    """Render a PSF with an 'inferno' colour map, nearest-neighbour."""
    p = psf / psf.max() if psf.max() > 0 else psf
    x = np.linspace(0, 1, len(_INFERNO))
    rgb = np.stack([np.interp(p, x, _INFERNO[:, c]) for c in range(3)], -1).astype(np.uint8)
    im = Image.fromarray(rgb, "RGB")
    s = max(1, size // max(psf.shape))
    return im.resize((psf.shape[1] * s, psf.shape[0] * s), Image.NEAREST)


def resize_max(img: np.ndarray, max_dim: int | None) -> np.ndarray:
    if max_dim is None or max(img.shape[:2]) <= max_dim:
        return img
    s = max_dim / max(img.shape[:2])
    w, h = int(round(img.shape[1] * s)), int(round(img.shape[0] * s))
    # high-quality downsizing with float precision via PIL on each channel
    chans = img[..., None] if img.ndim == 2 else img
    out = np.stack([np.asarray(Image.fromarray(chans[..., c].astype(np.float32), "F")
                               .resize((w, h), Image.LANCZOS)) for c in range(chans.shape[2])], -1)
    out = np.clip(out.astype(np.float64), 0, 1)
    return out[..., 0] if img.ndim == 2 else out


# ==========================================================================
# zoomable / pannable image viewer with split-compare
# ==========================================================================

class ImageViewer(tk.Canvas):
    MODES = ("Input", "Result", "Split", "Side by side")

    def __init__(self, master, **kw):
        super().__init__(master, background="#1e1e1e", highlightthickness=0, **kw)
        self.left = None      # PIL images
        self.right = None
        self.mode = "Input"
        self.zoom = 1.0
        self.cx = self.cy = 0.0   # image coordinates at viewport centre
        self.split = 0.5          # split position (fraction of width)
        self._drag = None
        self._tk = []
        self.on_zoom = None
        self.bind("<Configure>", lambda e: self.render())
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag", None))
        self.bind("<Double-Button-1>", lambda e: self.fit())
        self.bind("<MouseWheel>", self._wheel)                       # Windows / macOS
        self.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, 1.25))  # Linux
        self.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 0.8))
        self.bind("<Motion>", self._hover)

    # -- public ------------------------------------------------------------
    def set_images(self, left: np.ndarray | None, right: np.ndarray | None, refit=False):
        new_shape = None if left is None else left.shape[:2]
        old_shape = None if self.left is None else (self.left.height, self.left.width)
        self.left = None if left is None else to_pil(left)
        self.right = None if right is None else to_pil(right)
        if refit or new_shape != old_shape:
            self.fit()
        else:
            self.render()

    def set_mode(self, mode):
        self.mode = mode
        self.render()

    def fit(self):
        im = self._base()
        if im is None:
            self.render(); return
        W, H = self._pane_size()
        self.zoom = min(W / im.width, H / im.height, 8.0)
        self.cx, self.cy = im.width / 2, im.height / 2
        self.render()

    def actual_size(self):
        self.zoom = 1.0
        self.render()

    # -- internals -----------------------------------------------------------
    def _base(self):
        return self.left or self.right

    def _pane_size(self):
        W, H = max(self.winfo_width(), 50), max(self.winfo_height(), 50)
        if self.mode == "Side by side":
            W = W // 2 - 2
        return W, H

    def _crop(self, im, W, H):
        """Return (PIL image for a W x H viewport, x0, y0) at the current zoom."""
        z = self.zoom
        x0 = self.cx - W / 2 / z
        y0 = self.cy - H / 2 / z
        box = (x0, y0, x0 + W / z, y0 + H / z)
        # clamp crop to image, compute paste offset
        bx0, by0 = max(0, box[0]), max(0, box[1])
        bx1, by1 = min(im.width, box[2]), min(im.height, box[3])
        canvas = Image.new("RGB", (W, H), (30, 30, 30))
        if bx1 > bx0 and by1 > by0:
            region = im.crop((int(np.floor(bx0)), int(np.floor(by0)), int(np.ceil(bx1)), int(np.ceil(by1))))
            dw = max(1, int(round((np.ceil(bx1) - np.floor(bx0)) * z)))
            dh = max(1, int(round((np.ceil(by1) - np.floor(by0)) * z)))
            region = region.resize((dw, dh), Image.NEAREST if z >= 2 else Image.BILINEAR)
            px = int(round((np.floor(bx0) - box[0]) * z))
            py = int(round((np.floor(by0) - box[1]) * z))
            canvas.paste(region.convert("RGB"), (px, py))
        return canvas

    def render(self):
        self.delete("all")
        self._tk = []
        W, H = max(self.winfo_width(), 50), max(self.winfo_height(), 50)
        if self._base() is None:
            self.create_text(W / 2, H / 2, text="Open an image to begin  (Ctrl+O)",
                             fill="#9e9e9e", font=("Segoe UI", 14))
            return
        left = self.left
        right = self.right or self.left
        if self.mode == "Input" or self.right is None and self.mode != "Side by side":
            self._show(self._crop(left, W, H), 0, 0)
            self._label(8, 8, "Input" if self.mode != "Result" or self.right is None else "Result")
            if self.mode == "Result" and self.right is None:
                self._label(8, 30, "(no result yet - press Deblur)")
        elif self.mode == "Result":
            self._show(self._crop(right, W, H), 0, 0)
            self._label(8, 8, "Result")
        elif self.mode == "Split":
            a, b = self._crop(left, W, H), self._crop(right, W, H)
            sx = int(self.split * W)
            comp = b.copy()
            comp.paste(a.crop((0, 0, sx, H)), (0, 0))
            self._show(comp, 0, 0)
            self.create_line(sx, 0, sx, H, fill="white", width=2)
            self.create_oval(sx - 9, H / 2 - 9, sx + 9, H / 2 + 9, fill="white", outline="#555")
            self.create_text(sx, H / 2, text="<>", font=("Segoe UI", 7, "bold"))
            self._label(8, 8, "Input")
            self._label(W - 8, 8, "Result", anchor="ne")
        else:  # side by side
            pw = W // 2 - 2
            self._show(self._crop(left, pw, H), 0, 0)
            self._show(self._crop(right, pw, H), pw + 4, 0)
            self.create_line(pw + 2, 0, pw + 2, H, fill="#555", width=4)
            self._label(8, 8, "Input")
            self._label(pw + 12, 8, "Result" if self.right is not None else "Result (none yet)")
        if self.on_zoom:
            self.on_zoom(self.zoom)

    def _show(self, pil, x, y):
        ph = ImageTk.PhotoImage(pil)
        self._tk.append(ph)
        self.create_image(x, y, image=ph, anchor="nw")

    def _label(self, x, y, text, anchor="nw"):
        t = self.create_text(x + 6 if anchor == "nw" else x - 6, y + 3, text=text, anchor=anchor,
                             fill="white", font=("Segoe UI", 9, "bold"))
        bb = self.bbox(t)
        r = self.create_rectangle(bb[0] - 5, bb[1] - 2, bb[2] + 5, bb[3] + 2, fill="#202020", outline="#606060")
        self.tag_lower(r, t)

    def _press(self, e):
        W = self.winfo_width()
        if self.mode == "Split" and self.right is not None and abs(e.x - self.split * W) < 12:
            self._drag = ("split",)
        else:
            self._drag = ("pan", e.x, e.y, self.cx, self.cy)

    def _motion(self, e):
        if not self._drag:
            return
        if self._drag[0] == "split":
            self.split = min(max(e.x / max(self.winfo_width(), 1), 0.0), 1.0)
        else:
            _, x0, y0, cx, cy = self._drag
            self.cx = cx - (e.x - x0) / self.zoom
            self.cy = cy - (e.y - y0) / self.zoom
        self.render()

    def _hover(self, e):
        W = self.winfo_width()
        if self.mode == "Split" and self.right is not None and abs(e.x - self.split * W) < 12:
            self.configure(cursor="sb_h_double_arrow")
        else:
            self.configure(cursor="fleur" if self._base() is not None else "")

    def _wheel(self, e):
        self._zoom_at(e.x, e.y, 1.25 if e.delta > 0 else 0.8)

    def _zoom_at(self, x, y, f):
        if self._base() is None:
            return
        W, H = self._pane_size()
        if self.mode == "Side by side" and x > W:
            x -= W + 4
        # keep the image point under the cursor fixed
        ix = self.cx + (x - W / 2) / self.zoom
        iy = self.cy + (y - H / 2) / self.zoom
        self.zoom = float(np.clip(self.zoom * f, 0.05, 32))
        self.cx = ix - (x - W / 2) / self.zoom
        self.cy = iy - (y - H / 2) / self.zoom
        self.render()


# ==========================================================================
# main application
# ==========================================================================

class DeconvApp(tk.Tk):
    def __init__(self, path=None):
        super().__init__()
        self.title("Deconvolution Studio")
        try:
            self.iconbitmap(os.path.join(HERE, "app_icon.ico"))   # Windows window icon
        except Exception:
            pass
        self.geometry("1360x880")
        self.minsize(1000, 640)
        self._style()

        self.path = None
        self.full = None        # image as loaded (full res)
        self.original = None    # working-resolution copy of the loaded image
        self.reference = None   # sharp ground truth (after a test blur)
        self.observed = None    # image to be deblurred
        self.result = None
        self.psf = None
        self.psf_desc = ""
        self.psf_file = None
        self.jobs = queue.Queue()
        self.busy = False

        self._menu()
        self._layout()
        self._bind_keys()
        self.after(100, self._poll)

        start = path or os.path.join(HERE, "cameraman.png")
        if os.path.exists(start):
            self.after(200, lambda: self.load_image(start))

    # ------------------------------------------------------------------ look
    def _style(self):
        st = ttk.Style(self)
        for theme in ("vista", "clam"):
            if theme in st.theme_names():
                st.theme_use(theme)
                break
        st.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=6)
        st.configure("Head.TLabel", font=("Segoe UI", 10, "bold"))
        st.configure("Small.TLabel", font=("Segoe UI", 8), foreground="#555")
        st.configure("Metric.TLabel", font=("Consolas", 9))

    def _menu(self):
        m = tk.Menu(self)
        f = tk.Menu(m, tearoff=0)
        f.add_command(label="Open image...", accelerator="Ctrl+O", command=self.open_image)
        f.add_command(label="Load PSF file...", command=self.load_psf_file)
        f.add_separator()
        f.add_command(label="Save result...", accelerator="Ctrl+S", command=self.save_result)
        f.add_command(label="Save PSF...", command=self.save_psf)
        f.add_separator()
        f.add_command(label="Exit", command=self.destroy)
        m.add_cascade(label="File", menu=f)
        v = tk.Menu(m, tearoff=0)
        for i, mode in enumerate(ImageViewer.MODES):
            v.add_command(label=mode, accelerator=str(i + 1), command=lambda md=mode: self.set_view(md))
        v.add_separator()
        v.add_command(label="Fit to window", accelerator="0", command=lambda: self.viewer.fit())
        v.add_command(label="Actual size (100%)", command=lambda: self.viewer.actual_size())
        m.add_cascade(label="View", menu=v)
        h = tk.Menu(m, tearoff=0)
        h.add_command(label="Algorithm notes (ALGORITHMS.md)", command=lambda: self._open_doc("ALGORITHMS.md"))
        h.add_command(label="README", command=lambda: self._open_doc("README.md"))
        h.add_command(label="Quick guide", command=self.show_guide)
        m.add_cascade(label="Help", menu=h)
        self.config(menu=m)

    def _bind_keys(self):
        self.bind("<Control-o>", lambda e: self.open_image())
        self.bind("<Control-s>", lambda e: self.save_result())
        self.bind("<F5>", lambda e: self.deblur())
        for i, mode in enumerate(ImageViewer.MODES):
            self.bind(str(i + 1), lambda e, md=mode: self._key_view(e, md))
        self.bind("0", lambda e: self._key_view(e, None))

    def _key_view(self, e, mode):
        if isinstance(e.widget, (tk.Entry, ttk.Entry, ttk.Spinbox, ttk.Combobox)):
            return
        if mode is None:
            self.viewer.fit()
        else:
            self.set_view(mode)

    # --------------------------------------------------------------- layout
    def _layout(self):
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)

        # ---- left control panel (scrollable) ----
        side_outer = ttk.Frame(root, width=360)
        side_outer.pack(side="left", fill="y")
        side_outer.pack_propagate(False)
        self.side_footer = ttk.Frame(side_outer, padding=(10, 6))
        self.side_footer.pack(side="bottom", fill="x")
        ttk.Separator(side_outer).pack(side="bottom", fill="x")
        sc = tk.Canvas(side_outer, highlightthickness=0, width=342)
        sb = ttk.Scrollbar(side_outer, orient="vertical", command=sc.yview)
        side = ttk.Frame(sc, padding=(10, 8))
        side.bind("<Configure>", lambda e: sc.configure(scrollregion=sc.bbox("all")))
        sc.create_window((0, 0), window=side, anchor="nw", width=342)
        sc.configure(yscrollcommand=sb.set)
        sc.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        side.bind("<Enter>", lambda e: self.bind_all("<MouseWheel>", lambda ev: sc.yview_scroll(-int(ev.delta / 120), "units")))
        side.bind("<Leave>", lambda e: self.unbind_all("<MouseWheel>"))

        self._section_image(side)
        self._section_test(side)
        self._section_psf(side)
        self._section_deblur(side)

        # ---- right: toolbar + viewer + status ----
        right = ttk.Frame(root)
        right.pack(side="left", fill="both", expand=True)
        bar = ttk.Frame(right, padding=(6, 4))
        bar.pack(fill="x")
        ttk.Label(bar, text="View:").pack(side="left")
        self.view_var = tk.StringVar(value="Input")
        for i, mode in enumerate(ImageViewer.MODES):
            ttk.Radiobutton(bar, text=f"{mode}", value=mode, variable=self.view_var,
                            command=lambda: self.viewer.set_mode(self.view_var.get())).pack(side="left", padx=3)
        ttk.Button(bar, text="Fit", width=5, command=lambda: self.viewer.fit()).pack(side="right")
        ttk.Button(bar, text="100%", width=5, command=lambda: self.viewer.actual_size()).pack(side="right", padx=4)
        self.zoom_lbl = ttk.Label(bar, text="", width=8)
        self.zoom_lbl.pack(side="right")

        self.viewer = ImageViewer(right)
        self.viewer.pack(fill="both", expand=True)
        self.viewer.on_zoom = lambda z: self.zoom_lbl.configure(text=f"{z * 100:.0f}%")

        status = ttk.Frame(right, padding=(6, 3))
        status.pack(fill="x")
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status, textvariable=self.status_var).pack(side="left")
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=160)
        self.progress.pack(side="right")

    def _section_image(self, p):
        fr = ttk.LabelFrame(p, text=" 1  Image ", padding=8)
        fr.pack(fill="x", pady=(0, 8))
        row = ttk.Frame(fr); row.pack(fill="x")
        ttk.Button(row, text="Open image...", command=self.open_image).pack(side="left")
        ttk.Button(row, text="Test images", command=self._test_image_menu).pack(side="left", padx=6)
        self.file_lbl = ttk.Label(fr, text="no image", style="Small.TLabel", wraplength=300)
        self.file_lbl.pack(fill="x", pady=(6, 2))
        row = ttk.Frame(fr); row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="Working size:").pack(side="left")
        self.size_var = tk.StringVar(value="1024 px")
        cb = ttk.Combobox(row, textvariable=self.size_var, values=PREVIEW_SIZES, state="readonly", width=14)
        cb.pack(side="left", padx=6)
        cb.bind("<<ComboboxSelected>>", lambda e: self._rescale())
        ttk.Label(fr, text="Larger images are downsized for speed; choose\n'Full resolution' for the final result.",
                  style="Small.TLabel").pack(fill="x", pady=(4, 0))

    def _section_test(self, p):
        fr = ttk.LabelFrame(p, text=" 2  Test blur (optional) ", padding=8)
        fr.pack(fill="x", pady=(0, 8))
        ttk.Label(fr, text="Blur a sharp image to see how well it is restored.",
                  style="Small.TLabel").pack(fill="x")
        row = ttk.Frame(fr); row.pack(fill="x", pady=4)
        ttk.Label(row, text="Type:").pack(side="left")
        self.tb_type = tk.StringVar(value="Motion")
        cb = ttk.Combobox(row, textvariable=self.tb_type, values=["Gaussian", "Defocus disk", "Motion"],
                          state="readonly", width=14)
        cb.pack(side="left", padx=6)
        cb.bind("<<ComboboxSelected>>", lambda e: self._test_type_changed())
        self.tb_size = self._slider(fr, "Strength", 0.5, 30, 15, fmt="{:.1f}")
        self.tb_angle = self._slider(fr, "Angle (deg)", 0, 180, 30, fmt="{:.0f}")
        self.tb_noise = self._slider(fr, "Noise (%)", 0, 5, 0.5, fmt="{:.2f}")
        row = ttk.Frame(fr); row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="Apply test blur", command=self.apply_test_blur).pack(side="left")
        ttk.Button(row, text="Restore original", command=self.restore_original).pack(side="left", padx=6)

    def _section_psf(self, p):
        fr = ttk.LabelFrame(p, text=" 3  Point spread function ", padding=8)
        fr.pack(fill="x", pady=(0, 8))
        row = ttk.Frame(fr); row.pack(fill="x")
        ttk.Label(row, text="Source:").pack(side="left")
        self.psf_src = tk.StringVar(value=PSF_SOURCES[0])
        cb = ttk.Combobox(row, textvariable=self.psf_src, values=PSF_SOURCES, state="readonly", width=22)
        cb.pack(side="left", padx=6)
        cb.bind("<<ComboboxSelected>>", lambda e: self._psf_src_changed())
        self.psf_params = ttk.Frame(fr)
        self.psf_params.pack(fill="x", pady=4)
        self.psf_hint = ttk.Label(fr, text="", style="Small.TLabel", wraplength=300)
        self.psf_hint.pack(fill="x")
        row = ttk.Frame(fr); row.pack(fill="x", pady=(6, 4))
        self.btn_psf = ttk.Button(row, text="Get PSF", command=self.get_psf)
        self.btn_psf.pack(side="left")
        ttk.Button(row, text="Save PSF...", command=self.save_psf).pack(side="left", padx=6)
        pv = ttk.Frame(fr); pv.pack(fill="x")
        self.psf_canvas = tk.Canvas(pv, width=120, height=120, background="#000", highlightthickness=1,
                                    highlightbackground="#888")
        self.psf_canvas.pack(side="left")
        self.psf_info = ttk.Label(pv, text="no PSF yet", style="Metric.TLabel", wraplength=190, justify="left")
        self.psf_info.pack(side="left", padx=8, anchor="n")
        # parameter variables (created once, shown on demand)
        self.p_sigma = tk.DoubleVar(value=2.0)
        self.p_radius = tk.DoubleVar(value=5.0)
        self.p_length = tk.DoubleVar(value=15.0)
        self.p_angle = tk.DoubleVar(value=30.0)
        self.p_ksize = tk.IntVar(value=31)
        self.p_box = tk.IntVar(value=21)
        self._psf_src_changed()

    def _section_deblur(self, p):
        fr = ttk.LabelFrame(p, text=" 4  Deblur ", padding=8)
        fr.pack(fill="x", pady=(0, 8))
        row = ttk.Frame(fr); row.pack(fill="x")
        ttk.Label(row, text="Algorithm:").pack(side="left")
        self.alg = tk.StringVar(value=ALGORITHMS[0])
        cb = ttk.Combobox(row, textvariable=self.alg, values=ALGORITHMS, state="readonly", width=24)
        cb.pack(side="left", padx=6)
        cb.bind("<<ComboboxSelected>>", lambda e: self._alg_changed())
        self.reg = self._slider(fr, "Regularisation", -2, 2, 0, fmt=None,
                                show=lambda v: f"x{10 ** v:.2g}")
        self.iters = self._slider(fr, "Iterations", 5, 300, 30, fmt="{:.0f}", integer=True)
        self.alg_hint = ttk.Label(fr, text="", style="Small.TLabel", wraplength=300)
        self.alg_hint.pack(fill="x")
        ft = self.side_footer          # always visible, outside the scroll area
        self.btn_run = ttk.Button(ft, text="Deblur  (F5)", style="Accent.TButton", command=self.deblur)
        self.btn_run.pack(fill="x", pady=(2, 4))
        row = ttk.Frame(ft); row.pack(fill="x")
        ttk.Button(row, text="Save result...", command=self.save_result).pack(side="left")
        ttk.Button(row, text="Use result as input", command=self.result_as_input).pack(side="left", padx=6)
        self.metrics = ttk.Label(ft, text="", style="Metric.TLabel", justify="left")
        self.metrics.pack(fill="x", pady=(6, 0))
        self._alg_changed()

    def _slider(self, parent, label, lo, hi, init, fmt="{:.2f}", show=None, integer=False):
        row = ttk.Frame(parent); row.pack(fill="x", pady=1)
        ttk.Label(row, text=label, width=13).pack(side="left")
        var = tk.DoubleVar(value=init)
        val = ttk.Label(row, width=7, anchor="e")
        def upd(*_):
            v = var.get()
            if integer and v != round(v):
                var.set(round(v))       # re-triggers upd via the trace
                return
            val.configure(text=show(v) if show else fmt.format(v))
        var.trace_add("write", upd)     # also updates when set from code
        s = ttk.Scale(row, from_=lo, to=hi, variable=var)
        s.pack(side="left", fill="x", expand=True, padx=4)
        val.pack(side="left")
        upd()
        var._row = row
        return var

    def _spin(self, parent, label, var, lo, hi, inc):
        row = ttk.Frame(parent); row.pack(fill="x", pady=1)
        ttk.Label(row, text=label, width=16).pack(side="left")
        ttk.Spinbox(row, from_=lo, to=hi, increment=inc, textvariable=var, width=8).pack(side="left")

    # ----------------------------------------------------------- reactions
    def _test_type_changed(self):
        t = self.tb_type.get()
        st = "normal" if t == "Motion" else "disabled"
        for w in self.tb_angle._row.winfo_children():
            try:
                w.state(["!disabled"] if st == "normal" else ["disabled"])
            except Exception:
                pass

    def _psf_src_changed(self):
        for w in self.psf_params.winfo_children():
            w.destroy()
        src = self.psf_src.get()
        hints = {
            "Auto estimate": "Blind: decides between Gaussian, defocus and linear motion from the image spectrum, "
                             "then fits the parameters.",
            "Gaussian (manual)": "Soft optical blur. sigma in pixels.",
            "Defocus disk (manual)": "Out-of-focus blur. Radius of the blur disk in pixels.",
            "Motion (manual)": "Straight motion streak. Angle counter-clockwise from horizontal.",
            "Blind kernel": "Estimates an arbitrary kernel (e.g. curved camera shake). Kernel size must be larger "
                            "than the blur. Takes several seconds.",
            "Point sources": "Measures the PSF from stars / beads / small bright dots. Box must contain one whole "
                             "point image.",
            "Straight edge": "Measures a Gaussian PSF from the sharpest straight high-contrast edge in the image.",
            "From file": "Load a measured PSF (.npy, .txt, .csv or a grayscale image).",
        }
        self.psf_hint.configure(text=hints.get(src, ""))
        f = self.psf_params
        if src == "Gaussian (manual)":
            self._spin(f, "sigma (px)", self.p_sigma, 0.3, 30, 0.1)
        elif src == "Defocus disk (manual)":
            self._spin(f, "radius (px)", self.p_radius, 0.5, 60, 0.5)
        elif src == "Motion (manual)":
            self._spin(f, "length (px)", self.p_length, 1, 150, 0.5)
            self._spin(f, "angle (deg)", self.p_angle, 0, 180, 1)
        elif src == "Blind kernel":
            self._spin(f, "kernel size (px)", self.p_ksize, 9, 101, 2)
        elif src == "Point sources":
            self._spin(f, "box size (px)", self.p_box, 7, 101, 2)
        elif src == "From file":
            row = ttk.Frame(f); row.pack(fill="x")
            ttk.Button(row, text="Choose file...", command=self.load_psf_file).pack(side="left")
            ttk.Label(row, text=os.path.basename(self.psf_file) if self.psf_file else "(none)",
                      style="Small.TLabel").pack(side="left", padx=6)

    def _alg_changed(self):
        a = self.alg.get()
        hints = {
            "TV-ADMM (recommended)": "Edge-preserving total variation; handles image borders as unknown "
                                     "(no ringing). Best general choice.",
            "Richardson-Lucy": "Poisson maximum-likelihood; standard for astronomy / microscopy. "
                               "More iterations = sharper but noisier.",
            "RL-TV": "Richardson-Lucy with TV regularisation - can run many iterations without noise blow-up.",
            "Tikhonov": "Fast closed form with a gradient prior; weight chosen by cross-validation.",
            "Wiener": "Fastest closed-form filter; noise-to-signal ratio from a noise estimate.",
        }
        self.alg_hint.configure(text=hints[a] + "\nRegularisation: right = smoother, left = sharper.")
        st = ["!disabled"] if a in ("Richardson-Lucy", "RL-TV") else ["disabled"]
        for w in self.iters._row.winfo_children():
            try:
                w.state(st)
            except Exception:
                pass

    def set_view(self, mode):
        self.view_var.set(mode)
        self.viewer.set_mode(mode)

    def _test_image_menu(self):
        m = tk.Menu(self, tearoff=0)
        for name in ("cameraman.png", "astronaut.png"):
            p = os.path.join(HERE, name)
            if os.path.exists(p):
                m.add_command(label=name, command=lambda pp=p: self.load_image(pp))
        m.tk_popup(self.winfo_pointerx(), self.winfo_pointery())

    # ------------------------------------------------------------- file I/O
    def open_image(self):
        p = filedialog.askopenfilename(title="Open image", filetypes=IMG_TYPES)
        if p:
            self.load_image(p)

    def load_image(self, path):
        try:
            img = D.load_image(path)
        except Exception as e:
            messagebox.showerror("Open image", f"Could not read the image:\n{e}")
            return
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]
        if img.ndim == 3 and np.allclose(img[..., 0], img[..., 1]) and np.allclose(img[..., 1], img[..., 2]):
            img = img[..., 0]
        self.path, self.full = path, img
        self._rescale(initial=True)

    def _working_max(self):
        s = self.size_var.get()
        return None if s.startswith("Full") else int(s.split()[0])

    def _rescale(self, initial=False):
        if self.full is None:
            return
        self.original = resize_max(self.full, self._working_max())
        self.reference = None
        self.observed = self.original
        self.result = None
        h, w = self.full.shape[:2]
        wh, ww = self.original.shape[:2]
        kind = "gray" if self.full.ndim == 2 else "RGB"
        txt = f"{os.path.basename(self.path)}   {w}x{h} {kind}"
        if (wh, ww) != (h, w):
            txt += f"\nworking at {ww}x{wh}"
        self.file_lbl.configure(text=txt)
        self.metrics.configure(text="")
        self.viewer.set_images(self.observed, None, refit=True)
        self.set_view("Input")
        self.status_var.set("Image loaded" if initial else "Working size changed - PSF and result cleared")
        if not initial:
            self._set_psf(None, "")

    def load_psf_file(self):
        p = filedialog.askopenfilename(title="Load PSF", filetypes=[("PSF", "*.npy *.txt *.csv *.png *.tif *.tiff"),
                                                                    ("All files", "*.*")])
        if not p:
            return
        try:
            psf = D.load_psf(p)
        except Exception as e:
            messagebox.showerror("Load PSF", str(e))
            return
        self.psf_file = p
        self.psf_src.set("From file")
        self._psf_src_changed()
        self._set_psf(psf, f"file: {os.path.basename(p)}\n{psf.shape[1]}x{psf.shape[0]} px")

    def save_result(self):
        if self.result is None:
            messagebox.showinfo("Save result", "There is no result yet - press Deblur first.")
            return
        base = os.path.splitext(os.path.basename(self.path or "image"))[0]
        p = filedialog.asksaveasfilename(title="Save result", defaultextension=".png",
                                         initialfile=f"{base}_deblurred.png",
                                         filetypes=[("PNG", "*.png"), ("TIFF", "*.tif"), ("JPEG", "*.jpg")])
        if p:
            D.save_image(p, self.result)
            self.status_var.set(f"Saved {p}")

    def save_psf(self):
        if self.psf is None:
            messagebox.showinfo("Save PSF", "There is no PSF yet.")
            return
        p = filedialog.asksaveasfilename(title="Save PSF", defaultextension=".npy", initialfile="psf.npy",
                                         filetypes=[("NumPy array", "*.npy"), ("CSV", "*.csv"), ("PNG image", "*.png")])
        if not p:
            return
        if p.lower().endswith(".npy"):
            np.save(p, self.psf)
        elif p.lower().endswith(".csv"):
            np.savetxt(p, self.psf, delimiter=",")
        else:
            D.save_image(p, self.psf / self.psf.max())
        self.status_var.set(f"Saved PSF to {p}")

    def _open_doc(self, name):
        p = os.path.join(HERE, name)
        if not os.path.exists(p):
            messagebox.showinfo("Help", f"{name} not found next to app.py")
            return
        try:
            os.startfile(p)  # Windows
        except AttributeError:
            webbrowser.open("file://" + p)

    def show_guide(self):
        messagebox.showinfo("Quick guide", __doc__.split("Workflow")[1].strip("-\n "))

    # ---------------------------------------------------------- test blur
    def _test_psf(self):
        t, v = self.tb_type.get(), self.tb_size.get()
        if t == "Gaussian":
            return D.gaussian_psf(v), f"Gaussian sigma={v:.1f}"
        if t == "Defocus disk":
            return D.disk_psf(v), f"Disk R={v:.1f}"
        return D.motion_psf(v, self.tb_angle.get()), f"Motion L={v:.1f}, {self.tb_angle.get():.0f} deg"

    def apply_test_blur(self):
        if self.original is None:
            return
        psf, desc = self._test_psf()
        rng = np.random.default_rng(0)
        y = D.blur(self.original, psf) + rng.normal(0, self.tb_noise.get() / 100, self.original.shape)
        self.reference = self.original
        self.observed = np.clip(y, 0, 1)
        self.result = None
        self.true_psf, self.true_desc = psf, desc
        self.viewer.set_images(self.observed, None)
        self.set_view("Input")
        self.metrics.configure(text=f"Test blur: {desc}, noise {self.tb_noise.get():.2f}%\n"
                                    f"Blurred  PSNR {D.psnr(self.reference, self.observed):6.2f} dB  "
                                    f"SSIM {D.ssim(self.reference, self.observed):.3f}")
        self.status_var.set(f"Applied test blur ({desc}). Now get the PSF and deblur.")

    def restore_original(self):
        if self.original is None:
            return
        self.reference = None
        self.observed = self.original
        self.result = None
        self.metrics.configure(text="")
        self.viewer.set_images(self.observed, None)
        self.set_view("Input")
        self.status_var.set("Original restored")

    def result_as_input(self):
        if self.result is None:
            return
        self.observed = self.result
        self.result = None
        self.viewer.set_images(self.observed, None)
        self.set_view("Input")
        self.status_var.set("Result is now the input (iterative refinement)")

    # ----------------------------------------------------------------- PSF
    def _set_psf(self, psf, desc):
        self.psf, self.psf_desc = psf, desc
        self.psf_canvas.delete("all")
        if psf is None:
            self.psf_info.configure(text="no PSF yet")
            return
        im = psf_to_pil(psf, 118)
        self._psf_tk = ImageTk.PhotoImage(im)
        self.psf_canvas.create_image(60, 60, image=self._psf_tk)
        self.psf_info.configure(text=desc)

    def _compute_psf(self, img):
        """Runs in a worker thread.  Returns (psf, description)."""
        src = self.psf_src.get()
        if src == "Auto estimate":
            psf, info = D.estimate_psf(img, "auto")
            k = info["kind"]
            if k == "gaussian":
                d = f"auto: Gaussian\nsigma = {info['sigma']:.2f} px"
            elif k == "disk":
                d = f"auto: defocus disk\nradius = {info['radius']:.2f} px"
            else:
                d = f"auto: motion\nlength = {info['length']:.2f} px\nangle = {info['angle']:.1f} deg"
            return psf, d
        if src == "Gaussian (manual)":
            s = float(self.p_sigma.get()); return D.gaussian_psf(s), f"Gaussian\nsigma = {s:.2f} px"
        if src == "Defocus disk (manual)":
            r = float(self.p_radius.get()); return D.disk_psf(r), f"Defocus disk\nradius = {r:.2f} px"
        if src == "Motion (manual)":
            L, a = float(self.p_length.get()), float(self.p_angle.get())
            return D.motion_psf(L, a), f"Motion\nlength = {L:.1f} px\nangle = {a:.0f} deg"
        if src == "Blind kernel":
            k = int(self.p_ksize.get()) | 1
            psf = D.blind_estimate(img, size=k)
            return psf, f"blind kernel\n{k}x{k} px"
        if src == "Point sources":
            b = int(self.p_box.get()) | 1
            psf = D.estimate_from_point_source(img, size=b)
            return psf, f"from point sources\n{b}x{b} px"
        if src == "Straight edge":
            psf, s = D.estimate_from_edge(img)
            return psf, f"from edge: Gaussian\nsigma = {s:.2f} px"
        if src == "From file":
            if not self.psf_file:
                raise RuntimeError("Choose a PSF file first.")
            psf = D.load_psf(self.psf_file)
            return psf, f"file: {os.path.basename(self.psf_file)}"
        raise RuntimeError(src)

    def get_psf(self):
        if self.observed is None or self.busy:
            return
        img = self.observed
        self._start(f"Getting PSF ({self.psf_src.get()})...")

        def work():
            t = time.time()
            psf, d = self._compute_psf(img)
            return ("psf", psf, d, time.time() - t)
        self._run(work)

    # -------------------------------------------------------------- deblur
    def deblur(self):
        if self.observed is None or self.busy:
            return
        img = self.observed
        need_psf = self.psf is None
        alg, mult, iters = self.alg.get(), 10 ** self.reg.get(), int(self.iters.get())
        psf0 = self.psf
        self._start("Estimating PSF and deblurring..." if need_psf else f"Deblurring with {alg.split(' (')[0]}...")

        def work():
            t = time.time()
            psf, d = (self._compute_psf(img) if need_psf else (psf0, None))
            g = to_gray(img)
            if alg.startswith("TV"):
                lam = 0.25 * gcv_lambda(edgetaper(g, psf), psf) * mult
                out, par = D.deconvolve(img, psf, "tv", lam=lam), f"lambda={lam:.2e}"
            elif alg == "Richardson-Lucy":
                out, par = D.deconvolve(img, psf, "rl", iterations=iters), f"{iters} iterations"
            elif alg == "RL-TV":
                tv = 0.003 * mult
                out, par = D.deconvolve(img, psf, "rl_tv", iterations=iters, tv=tv), f"{iters} it, tv={tv:.1e}"
            elif alg == "Tikhonov":
                lam = gcv_lambda(edgetaper(g, psf), psf) * mult
                out, par = D.deconvolve(img, psf, "tikhonov", lam=lam), f"lambda={lam:.2e}"
            else:
                nsr = max(60 * estimate_noise_sigma(g) ** 2, 1e-4) * mult
                out, par = D.deconvolve(img, psf, "wiener", nsr=nsr), f"NSR={nsr:.2e}"
            return ("result", out, psf, d, f"{alg.split(' (')[0]}: {par}", time.time() - t)
        self._run(work)

    # ------------------------------------------------------ worker plumbing
    def _start(self, msg):
        self.busy = True
        self.status_var.set(msg)
        self.progress.start(12)
        for b in (self.btn_run, self.btn_psf):
            b.state(["disabled"])
        self.configure(cursor="watch")

    def _run(self, fn):
        def target():
            try:
                self.jobs.put(("ok", fn()))
            except Exception as e:
                self.jobs.put(("err", e, traceback.format_exc()))
        threading.Thread(target=target, daemon=True).start()

    def _poll(self):
        try:
            while True:
                job = self.jobs.get_nowait()
                self._finish(job)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _finish(self, job):
        self.busy = False
        self.progress.stop()
        for b in (self.btn_run, self.btn_psf):
            b.state(["!disabled"])
        self.configure(cursor="")
        if job[0] == "err":
            self.status_var.set("Error - see message")
            messagebox.showerror("Error", f"{job[1]}\n\n{job[2][-800:]}")
            return
        payload = job[1]
        if payload[0] == "psf":
            _, psf, d, dt = payload
            self._set_psf(psf, d)
            self.status_var.set(f"PSF ready ({dt:.1f} s). Press Deblur.")
        else:
            _, out, psf, d, par, dt = payload
            if d is not None:
                self._set_psf(psf, d)
            self.result = out
            self.viewer.set_images(self.observed, out)
            if self.view_var.get() == "Input":
                self.set_view("Split")
            msg = f"{par}\ntime {dt:.1f} s"
            if self.reference is not None and self.reference.shape == out.shape:
                msg = (self.metrics.cget("text").split("\nResult")[0] + "\n"
                       f"Result   PSNR {D.psnr(self.reference, out):6.2f} dB  SSIM {D.ssim(self.reference, out):.3f}"
                       f"\n{msg}")
            self.metrics.configure(text=msg)
            self.status_var.set(f"Done in {dt:.1f} s - drag the divider to compare, Ctrl+S to save")


def _report_crash():
    """Write the traceback to startup_error.log and show it in a dialog."""
    tb = traceback.format_exc()
    log = os.path.join(APP_DIR, "startup_error.log")
    try:
        with open(log, "w", encoding="utf-8") as f:
            f.write(tb)
    except OSError:
        pass
    print(tb, file=sys.stderr)
    try:
        r = tk.Tk(); r.withdraw()
        messagebox.showerror("Deconvolution Studio - error", tb[-1500:] + f"\n\nSaved to {log}")
        r.destroy()
    except Exception:
        pass


def selftest():
    """`DeconvolutionStudio.exe --selftest`: checks the bundled libraries and the
    GUI toolkit, writes selftest.txt next to the exe and exits."""
    out = os.path.join(APP_DIR, "selftest.txt")
    lines = []
    try:
        x = D.load_image(os.path.join(HERE, "cameraman.png"), gray=True)[128:384, 128:384]
        y = D.blur(x, D.motion_psf(11, 30)) + np.random.default_rng(0).normal(0, 0.005, x.shape)
        psf, info = D.estimate_psf(y, "auto")
        r = D.deconvolve(y, psf, "tv")
        lines.append(f"PSF estimate: {info['kind']} {info.get('length', ''):} {info.get('angle', '')}")
        lines.append(f"PSNR blurred {D.psnr(x, y):.2f} dB -> deblurred {D.psnr(x, r):.2f} dB")
        app = DeconvApp(os.path.join(HERE, "cameraman.png"))
        app.after(1500, app.destroy)
        app.mainloop()
        lines.append("GUI: ok")
        lines.append("SELFTEST PASSED")
        code = 0
    except Exception:
        lines.append(traceback.format_exc())
        lines.append("SELFTEST FAILED")
        code = 1
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(code)


def main():
    if "--selftest" in sys.argv:
        selftest()
    try:
        app = DeconvApp(sys.argv[1] if len(sys.argv) > 1 else None)
        app.report_callback_exception = lambda *a: messagebox.showerror(
            "Error", "".join(traceback.format_exception(*a))[-1500:])
        app.mainloop()
    except Exception:
        _report_crash()
        sys.exit(1)


if __name__ == "__main__":
    main()
