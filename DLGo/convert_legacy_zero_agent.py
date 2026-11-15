"""Convert a zero agent trained before the TF2 port so it runs on TF 2.10.

Old agents (e.g. bots/zeroagent_v2.hdf5) were saved by standalone Keras 2.3.1
with channels-first convolutions, which TensorFlow cannot run on the CPU. This
rebuilds the same network in the current channels-last layout (with
legacy_heads=True so the heads compute exactly what they computed before) and
copies the trained weights over by layer name.

Usage:
    python convert_legacy_zero_agent.py bots/zeroagent_v2.hdf5 bots/zeroagent_v2_tf2.hdf5
"""
import argparse
import os
import tempfile

import h5py

from dlgo import encoders
from dlgo import networks
from dlgo import zero


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    args = parser.parse_args()

    with h5py.File(args.input_file, 'r') as f:
        encoder_name = f['encoder'].attrs['name']
        if not isinstance(encoder_name, str):
            encoder_name = encoder_name.decode('ascii')
        board_size = int(f['encoder'].attrs['board_size'])
        weights = f['model']['kerasmodel']['model_weights']
        layer_names = [n.decode('utf8') if isinstance(n, bytes) else n
                       for n in weights.attrs['layer_names']]
        weighted = [n for n in layer_names
                    if len(weights[n].attrs['weight_names']) > 0]
        blocks = sum(1 for n in layer_names if n.startswith('add_'))
        # Copy the Keras model group out to a standalone file for load_weights.
        fd, tmp = tempfile.mkstemp(suffix='.h5')
        os.close(fd)
        with h5py.File(tmp, 'w') as g:
            root = f['model']['kerasmodel']
            for k, v in root.attrs.items():
                g.attrs[k] = v
            for k in root.keys():
                f.copy(root[k], g, k)

    encoder = encoders.get_encoder_by_name(encoder_name, board_size)
    model = networks.dual_residual_network(
        input_shape=encoder.shape(), blocks=blocks, legacy_heads=True)
    new_weighted = [l.name for l in model.layers if l.weights]
    missing = sorted(set(weighted) ^ set(new_weighted))
    if missing:
        os.unlink(tmp)
        raise SystemExit('Layer mismatch between old and new model: %s' % missing)
    try:
        # by_name matching; raises on any shape mismatch.
        model.load_weights(tmp, by_name=True)
    finally:
        os.unlink(tmp)

    agent = zero.ZeroAgent(model, encoder)
    with h5py.File(args.output_file, 'w') as outf:
        agent.serialize(outf)
    print('Converted %s -> %s (%d residual blocks, %d weighted layers)' % (
        args.input_file, args.output_file, blocks, len(new_weighted)))


if __name__ == '__main__':
    main()
