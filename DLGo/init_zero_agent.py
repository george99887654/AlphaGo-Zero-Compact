import argparse

import h5py

from keras.layers import Activation, BatchNormalization
from keras.layers import Conv2D, Dense, Flatten, Input
from keras.models import Model
import dlgo.networks
from dlgo import zero
from dlgo import encoders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--board-size', type=int, default=19)
#     parser.add_argument('--network', default='large')
#     parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('output_file')
    args = parser.parse_args()

    encoder = encoders.get_encoder_by_name('zero', args.board_size)
    board_input = Input(shape=encoder.shape(), name='board_input')

    pb = board_input
    for i in range(4):                     # <1>
        pb = Conv2D(64, (3, 3),            # <1>
            padding='same',                # <1>
            data_format='channels_first',  # <1>
            activation='relu')(pb)         # <1>

    policy_conv = \
        Conv2D(2, (1, 1),                                   # <2>
            data_format='channels_first',                   # <2>
            activation='relu')(pb)                          # <2>
    policy_flat = Flatten()(policy_conv)                    # <2>
    policy_output = \
        Dense(encoder.num_moves(), activation='softmax')(   # <2>
            policy_flat)                                    # <2>

    value_conv = \
        Conv2D(1, (1, 1),                                    # <3>
            data_format='channels_first',                    # <3>
            activation='relu')(pb)                           # <3>
    value_flat = Flatten()(value_conv)                       # <3>
    value_hidden = Dense(256, activation='relu')(value_flat) # <3>
    value_output = Dense(1, activation='tanh')(value_hidden) # <3>

    model = Model(
        inputs=[board_input],
        outputs=[policy_output, value_output])

    model.summary()

    new_agent = zero.ZeroAgent(model, encoder, rounds_per_move=1000, c=2.0)
    with h5py.File(args.output_file, 'w') as outf:
        new_agent.serialize(outf)


if __name__ == '__main__':
    main()