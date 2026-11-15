from dlgo.cuda_paths import add_nvidia_dll_dirs as _add_nvidia_dll_dirs

# Must happen before TensorFlow is imported (see cuda_paths.py).
_add_nvidia_dll_dirs()
