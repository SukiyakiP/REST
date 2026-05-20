# Runtime hook: pre-load torch CUDA DLLs before torch's _load_dll_libraries() runs.
#
# torch/_load_dll_libraries() on Windows uses LoadLibraryExW with
# LOAD_LIBRARY_SEARCH_USER_DIRS | LOAD_LIBRARY_SEARCH_SYSTEM32 flags.
# When that call returns error 1114 (DLL_INIT_FAILED) — not 126 (not found) —
# it raises immediately without a fallback.  Pre-loading the DLLs via
# ctypes.CDLL(full_path) uses LoadLibraryW (PATH-aware), and the Windows DLL
# cache means the subsequent LoadLibraryExW call skips re-initialization.

import ctypes
import os
import sys

if sys.platform == "win32" and getattr(sys, "frozen", False):
    _torch_lib = os.path.join(sys._MEIPASS, "torch", "lib")

    if os.path.isdir(_torch_lib):
        # Register directory for subsequent implicit loads
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_torch_lib)

        # Also add to PATH so LoadLibraryW can find transitive deps
        os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")

        # Load CUDA deps in dependency order; errors are non-fatal here —
        # torch will raise a clearer message if something critical is missing.
        _load_order = [
            "libiomp5md.dll",
            "cudart64_12.dll",
            "cublas64_12.dll",
            "cublasLt64_12.dll",
            "cufft64_11.dll",
            "curand64_10.dll",
            "cusparse64_12.dll",
            "cudnn64_9.dll",
            "c10.dll",
            "c10_cuda.dll",
            "torch_cpu.dll",
            "torch_cuda.dll",
            "torch.dll",
        ]
        for _dll in _load_order:
            _fp = os.path.join(_torch_lib, _dll)
            if os.path.exists(_fp):
                try:
                    ctypes.CDLL(_fp)
                except OSError:
                    pass
