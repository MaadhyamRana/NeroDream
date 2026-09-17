# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the NeroDream desktop GUI.

Bundles gui.py plus every cached model weight (~/.cache/torch/hub/checkpoints)
so the packaged app runs fully offline -- no torch hub download on first
launch. Run `python scripts/prefetch_models.py` first if that cache is
missing any model.

Build:
pyinstaller nerodream.spec

Output lands in dist/NeroDream/, a onedir build with a folder containing
NeroDream.exe plus its dependencies and bundled weights.
"""
import os
from pathlib import Path

import torch
from PyInstaller.utils.hooks import collect_all

checkpoints_dir = Path(torch.hub.get_dir()) / "checkpoints"
weight_files = sorted(checkpoints_dir.glob("*.pth")) if checkpoints_dir.is_dir() else []
if not weight_files:
    raise SystemExit(
        f"No cached model weights found in {checkpoints_dir}. "
        "Run `python scripts/prefetch_models.py` first."
    )

datas = [(str(path), "torch_home/hub/checkpoints") for path in weight_files]
datas.append(("images/logo.jpg", "images"))

# PyInstaller's bundled torchvision hook only knows about the older `_C.so`
# extension name and misses newer torchvision's `_C_stable.so`, silently
# dropping the native ops extension (breaks at runtime with
# "operator torchvision::nms does not exist"). collect_all() finds it by
# scanning the installed package directly instead of relying on that hook.
tv_datas, tv_binaries, tv_hiddenimports = collect_all("torchvision")
datas += tv_datas

a = Analysis(
    ["gui.py"],
    pathex=[],
    binaries=tv_binaries,
    datas=datas,
    hiddenimports=tv_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NeroDream",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon="images/logo.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="NeroDream",
)
