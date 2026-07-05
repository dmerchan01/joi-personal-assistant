#!/usr/bin/env bash
# OBSOLETE — joi/cuda_libs.py now preloads libcudnn/libcublas from Python,
# so no shell setup is needed. Kept only as a bash-shell fallback.
# (Note: bash syntax — cannot be sourced from fish.)
# Note: __path__ (not __file__) — these are namespace packages, so __file__ is None.
export LD_LIBRARY_PATH="$(python -c 'import nvidia.cublas.lib, nvidia.cudnn.lib; print(nvidia.cublas.lib.__path__[0] + ":" + nvidia.cudnn.lib.__path__[0])'):${LD_LIBRARY_PATH}"
echo "LD_LIBRARY_PATH set for this shell."
