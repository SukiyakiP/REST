# PyInstaller Packaging — REST Inference GUI

**Environment**: Python 3.14.2, PyInstaller 6.19.0, torch 2.10.0+cu126, conda env `REST`  
**Output**: `dist/REST_Inference_GUI/` — onedir bundle, no console window

---

## Which build to use?

| | CUDA build | CPU build |
|---|---|---|
| Script | `Inference_GUI.py` | `Inference_GUI_CPU.py` |
| Spec | `Inference_GUI.spec` | `Inference_GUI_CPU.spec` |
| Conda env | `REST` | `REST_CPU` |
| Output folder | `dist/REST_Inference_GUI/` | `dist/REST_Inference_GUI_CPU/` |
| Disk size | ~4.2 GB | ~0.6 GB |
| RAM at runtime | ~800 MB | ~450 MB |
| Requires GPU | No (falls back to CPU) | No (always CPU) |

**Use the CPU build for distribution.** It's 7× smaller and inference is fast enough on CPU for this model size.

---

## Quick Build (normal rebuild)

**CPU build (recommended for distribution):**
```powershell
Stop-Process -Name "REST_Inference_GUI_CPU" -Force -ErrorAction SilentlyContinue
Remove-Item "M:\Alex\Python\REST\dist\REST_Inference_GUI_CPU" -Recurse -Force -ErrorAction SilentlyContinue
cd M:\Alex\Python\REST
conda run -n REST_CPU python -m PyInstaller Inference_GUI_CPU.spec --noconfirm
```

**CUDA build (for GPU machines):**
```powershell
Stop-Process -Name "REST_Inference_GUI" -Force -ErrorAction SilentlyContinue
Remove-Item "M:\Alex\Python\REST\dist\REST_Inference_GUI" -Recurse -Force -ErrorAction SilentlyContinue
cd M:\Alex\Python\REST
python -m PyInstaller Inference_GUI.spec --noconfirm
```

> Always kill any running instance and remove the dist folder before rebuilding — PyInstaller fails with `FileExistsError` if the folder is locked.  
> Use `python -m PyInstaller` not bare `pyinstaller` — the conda env's PATH sometimes resolves the wrong binary.  
> For the CPU build, use `conda run -n REST_CPU` so PyInstaller picks up the CPU-only torch from the right env.

---

## Files That Make It Work

### `Inference_GUI.spec`
The spec handles three non-obvious things:

**1. MNE .pyi stubs** — MNE uses `lazy_loader`, which reads `.pyi` stub files to enumerate lazy imports. These are not Python modules, so PyInstaller skips them by default. The spec globs all 21 of them explicitly:

```python
_mne_root = os.path.dirname(__import__('mne').__file__)
_mne_pyi = []
for _fp in glob.glob(os.path.join(_mne_root, '**', '*.pyi'), recursive=True):
    _rel = os.path.relpath(os.path.dirname(_fp), os.path.dirname(_mne_root))
    _mne_pyi.append((_fp, _rel))
# then: datas=[...] + _mne_pyi
```

Without this: `ValueError: Cannot load imports from non-existent stub 'mne/__init__.pyi'`

**2. Bundled model weights + artifact params**

```python
datas=[
    ('model_artifact.pth',   '.'),   # → _internal/model_artifact.pth
    ('artifact_params.json', '.'),   # → _internal/artifact_params.json
]
```

`model_artifact.pth` is a copy of whichever checkpoint you want to ship (currently `checkpoints/20260514_1309_w120_artrepeat2/best_acc.pth`). Copy a new checkpoint over this file before rebuilding to update the bundled model.

**3. `pydoc` must NOT be in excludes** — `scipy._lib._docscrape` imports `pydoc` at load time. Excluding it causes an immediate crash on startup. Keep it out of the `excludes` list.

---

### `rthook_torch_dll.py` — CUDA DLL pre-loader

This is the most critical fix. torch's `_load_dll_libraries()` on Windows uses `LoadLibraryExW` with `LOAD_LIBRARY_SEARCH_USER_DIRS | LOAD_LIBRARY_SEARCH_SYSTEM32` flags. If any CUDA DLL returns error 1114 (DLL_INIT_FAILED) rather than 126 (not found), it raises immediately with no fallback:

```
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed.
Error loading "..._internal\torch\lib\c10.dll"
```

The DLLs ARE present — this is not a missing file problem. The fix is to pre-load them via `ctypes.CDLL(full_path)` (which uses PATH-aware `LoadLibraryW`) before torch's strict loader runs. Windows DLL cache means torch's subsequent `LoadLibraryExW` call finds them already loaded and skips re-initialization.

The hook runs before any app code via `runtime_hooks=['rthook_torch_dll.py']` in the spec.

Load order matters — dependencies must be loaded before dependents:
```
libiomp5md → cudart → cublas → cublasLt → cufft → curand → cusparse → cudnn
→ c10 → c10_cuda → torch_cpu → torch_cuda → torch
```

---

### `ArtifactFilter.py` — frozen path resolution

`ArtifactFilter.py` loads `artifact_params.json`. It handles both frozen (exe) and script modes:

```python
def _find_params_file():
    if getattr(sys, 'frozen', False):
        beside_exe = os.path.join(os.path.dirname(sys.executable), "artifact_params.json")
        if os.path.exists(beside_exe):
            return beside_exe                       # user dropped one next to exe
        return os.path.join(sys._MEIPASS, "artifact_params.json")  # bundled fallback
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifact_params.json")
```

In onedir mode, `sys._MEIPASS` points to `_internal/`, not the folder containing the `.exe`. The params file is in `_internal/` (bundled), but a user can override it by dropping a `artifact_params.json` next to the exe.

---

## Updating the Bundled Model

1. Copy the new checkpoint to `M:\Alex\Python\REST\model_artifact.pth`
2. Rebuild with `python -m PyInstaller Inference_GUI.spec --noconfirm`

`Inference_GUI.py` reads the model from `sys._MEIPASS/model_artifact.pth` when frozen, so the name `model_artifact.pth` is hardcoded in both the spec and the GUI.

---

## Diagnosing a Startup Crash

Switch `console=False` → `console=True` in the spec, rebuild, and run from a terminal. The Python traceback will appear in the console window.

```python
# In Inference_GUI.spec, EXE block:
console=True,   # temporary — switch back to False for release
```

Alternatively, redirect stderr when launching:
```powershell
Start-Process ".\REST_Inference_GUI.exe" -RedirectStandardError "err.txt"
Start-Sleep 10
Get-Content err.txt
```

---

## MNE Submodules — Permanent Fix

MNE has many submodules that are lazily imported or only triggered during EDF reading. Rather than adding them one by one, the spec now uses `collect_submodules('mne')` which bundles every MNE submodule at once:

```python
from PyInstaller.utils.hooks import collect_submodules
# in Analysis:
hiddenimports=collect_submodules('mne') + [ ...other imports... ]
```

The `.pyi` stub glob in `datas` is still required separately — `collect_submodules` handles Python modules but not data files.

---

## Known Harmless Warnings During Build

These appear in build output and can be ignored:

- `WARNING: Failed to collect submodules for 'torch.utils.tensorboard'` — tensorboard not installed, not needed
- `WARNING: Hidden import 'sklearn.neighbors._typedefs' not found` — sklearn internal; safe to ignore
- `WARNING: scipy.special._cdflib not found` — scipy internal; safe to ignore
- Conda `liblapack / libcblas / libblas` warnings — pip-installed numpy, not a problem
- `pyqtgraph.opengl` warning — OpenGL not installed, pyqtgraph still works without it

---

## Distribution

Zip `dist/REST_Inference_GUI/` in its entirety. The recipient needs no Python, no conda, no CUDA toolkit installed — all DLLs are inside `_internal/torch/lib/`. The folder is ~4.2 GB due to CUDA.
