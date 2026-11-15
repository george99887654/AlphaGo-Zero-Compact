"""Make pip-installed NVIDIA CUDA/cuDNN libraries visible to TensorFlow.

The `nvidia-*-cu11` wheels (see requirements-gpu.txt) put their DLLs in
site-packages/nvidia/<lib>/bin, which Windows does not search. TensorFlow loads
CUDA libraries through the normal DLL search path, so those folders are added
to PATH here. This must run before TensorFlow initializes the GPU; it is
called from dlgo/__init__.py, so import dlgo before tensorflow.

It does nothing when the wheels are not installed (CPU-only setups) or on
non-Windows systems, where TensorFlow finds the libraries itself.
"""
import glob
import os
import site
import sys


def add_nvidia_dll_dirs():
    if sys.platform != 'win32':
        return []
    roots = list(site.getsitepackages())
    try:
        roots.append(site.getusersitepackages())
    except AttributeError:
        pass
    dirs = []
    for root in roots:
        dirs.extend(sorted(glob.glob(os.path.join(root, 'nvidia', '*', 'bin'))))
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        return []
    path = os.environ.get('PATH', '')
    missing = [d for d in dirs if d not in path.split(os.pathsep)]
    if missing:
        os.environ['PATH'] = os.pathsep.join(missing + [path])
    for d in dirs:
        try:
            os.add_dll_directory(d)
        except (AttributeError, OSError):
            pass
    return dirs
