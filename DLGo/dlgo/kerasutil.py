from __future__ import absolute_import
import tempfile
import os

import h5py
from tensorflow import keras
from tensorflow.keras.models import load_model, save_model


def save_model_to_hdf5_group(model, f):
    # Use Keras save_model to save the full model (including optimizer
    # state) to a file.
    # Then we can embed the contents of that HDF5 file inside ours.
    tempfd, tempfname = tempfile.mkstemp(prefix='tmp-kerasmodel')
    try:
        os.close(tempfd)
        save_model(model, tempfname, save_format='h5')
        serialized_model = h5py.File(tempfname, 'r')
        root_item = serialized_model.get('/')
        serialized_model.copy(root_item, f, 'kerasmodel')
        serialized_model.close()
    finally:
        os.unlink(tempfname)


def load_model_from_hdf5_group(f, custom_objects=None):
    # Extract the model into a temporary file. Then we can use Keras
    # load_model to read it.
    tempfd, tempfname = tempfile.mkstemp(prefix='tmp-kerasmodel')
    try:
        os.close(tempfd)
        serialized_model = h5py.File(tempfname, 'w')
        root_item = f.get('kerasmodel')
        for attr_name, attr_value in root_item.attrs.items():
            serialized_model.attrs[attr_name] = attr_value
        for k in root_item.keys():
            f.copy(root_item.get(k), serialized_model, k)
        serialized_model.close()
        # compile=False: loaded models are used for inference or recompiled
        # by ZeroAgent.train, so the saved optimizer state is never needed,
        # and files saved before any training (gen 0 / init) have none,
        # which made Keras warn "No training configuration found".
        return load_model(tempfname, custom_objects=custom_objects,
                          compile=False)
    finally:
        os.unlink(tempfname)


def set_gpu_memory_target(frac):
    """Configure Tensorflow to use a fraction of available GPU memory.
    Use this for evaluating models in parallel. By default, Tensorflow
    will try to map all available GPU memory in advance. You can
    configure to use just a fraction so that multiple processes can run
    in parallel. For example, if you want to use 2 works, set the
    memory fraction to 0.5.
    If you are using Python multiprocessing, you must call this function
    from the *worker* process (not from the parent).
    This function does nothing if Keras is using a backend other than
    Tensorflow.
    """
    # TF2: let each process grow its GPU memory on demand so several
    # worker processes can share one GPU. `frac` is kept for API
    # compatibility. Must run before TensorFlow touches the GPU.
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # GPU already initialized in this process; keep current config.
            pass
