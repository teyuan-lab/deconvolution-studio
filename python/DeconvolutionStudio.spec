# PyInstaller spec for Deconvolution Studio  ->  dist/DeconvolutionStudio.exe
# Build with:  build_exe.bat   (or:  pyinstaller DeconvolutionStudio.spec)
import os

datas = [(f, ".") for f in ("cameraman.png", "astronaut.png", "README.md", "ALGORITHMS.md",
                            "app_icon.ico") if os.path.exists(f)]

a = Analysis(
    ["app.py"],
    pathex=["."],
    datas=datas,
    hiddenimports=["PIL._tkinter_finder", "scipy.special._cdflib", "scipy.optimize", "scipy.ndimage"],
    excludes=["matplotlib", "pandas", "IPython", "jupyter", "pytest", "sympy", "PyQt5", "PySide6",
              "notebook", "sphinx", "numba", "torch", "tensorflow"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="DeconvolutionStudio",
    icon="app_icon.ico" if os.path.exists("app_icon.ico") else None,
    console=False,          # windowed app, no console
    upx=False,
    runtime_tmpdir=None,
)
