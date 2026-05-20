# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for REST Inference GUI — CPU-only build
# Build using the REST_CPU conda environment (CPU-only torch, ~800 MB output).
# Activate REST_CPU first: conda activate REST_CPU
# Then: python -m PyInstaller Inference_GUI_CPU.spec --noconfirm

import os
import glob as _glob
from PyInstaller.utils.hooks import collect_submodules
block_cipher = None

# MNE uses lazy_loader which requires .pyi stub files at runtime.
# Collect all .pyi files from the mne package and bundle them as data.
_mne_root = os.path.join(os.path.dirname(__import__('mne').__file__))
_mne_pyi = []
for _fp in _glob.glob(os.path.join(_mne_root, '**', '*.pyi'), recursive=True):
    _rel = os.path.relpath(os.path.dirname(_fp), os.path.dirname(_mne_root))
    _mne_pyi.append((_fp, _rel))

a = Analysis(
    ['Inference_GUI_CPU.py'],
    pathex=[r'M:\Alex\Python\REST'],
    binaries=[],
    datas=[
        ('model_artifact.pth',   '.'),
        ('artifact_params.json', '.'),
    ] + _mne_pyi,
    hiddenimports=collect_submodules('mne') + [
        # scipy
        'scipy.signal',
        'scipy.io',
        'scipy.io.matlab',
        'scipy.ndimage',
        'scipy.stats',
        'scipy.sparse',
        'scipy.sparse.linalg',
        # sklearn (MNE pulls this in)
        'sklearn',
        'sklearn.utils',
        'sklearn.utils._cython_blas',
        'sklearn.neighbors._typedefs',
        'sklearn.neighbors._quad_tree',
        'sklearn.tree._utils',
        # matplotlib
        'matplotlib',
        'matplotlib.backends.backend_tkagg',
        'matplotlib.backends._backend_tk',
        # torch
        'torch',
        'torch.nn',
        'torch.nn.functional',
        'torch.utils.data',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],   # No CUDA DLL pre-loading needed for CPU torch
    excludes=[
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'setuptools',
        'pkg_resources',
        'pywin32',
        'win32com',
        'xmlrpc',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='REST_Inference_GUI_CPU',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='REST_Inference_GUI_CPU',
)
