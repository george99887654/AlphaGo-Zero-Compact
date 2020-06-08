import argparse

import h5py

from keras.layers import Activation, BatchNormalization
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.models import Model
import dlgo.networks
from dlgo import zero
from dlgo import encoders
from dlgo import networks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', type=int, default=19)
#     parser.add_argument('--network', default='large')
#     parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('output_file')
    args = parser.parse_args()

    encoder = encoders.get_encoder_by_name('zero', args.board_size)
    model = networks.dual_residual_network(input_shape = encoder.shape(), blocks=8)
    model.summary()

    new_agent = zero.ZeroAgent(model, encoder, rounds_per_move=1000, c=2.0)
    with h5py.File(args.output_file, 'w') as outf:
        new_agent.serialize(outf)


if __name__ == '__main__':
    main()