"""Preload pip-installed CUDA libraries (nvidia-cublas-cu12, nvidia-cudnn-cu12).

ctranslate2 dlopens libcublas.so.12 / libcudnn.so.9 by name, which normally
requires LD_LIBRARY_PATH to point into the nvidia wheels. Loading them here
with RTLD_GLOBAL makes them resolvable without any shell setup.
"""
import ctypes
import glob
import importlib.util
import os

_PACKAGES = ("nvidia.cublas.lib", "nvidia.cudnn.lib")


def preload() -> None:
    for pkg in _PACKAGES:
        spec = importlib.util.find_spec(pkg)
        if spec is None or not spec.submodule_search_locations:
            continue  # wheel not installed; fall back to system libs
        for path in spec.submodule_search_locations:
            for lib in sorted(glob.glob(os.path.join(path, "lib*.so.*"))):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
