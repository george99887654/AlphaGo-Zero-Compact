from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

'''Network inputs are channels-first board tensors (planes, N, N), as produced
by the zero encoder. Internally every model permutes them to channels-last
(N, N, planes) first, because TensorFlow only supports channels-last
convolutions on the CPU (channels-last is also the fast layout on GPU).
'''


def to_channels_last():
    return Permute((2, 3, 1), name="to_channels_last")


def num_moves(input_shape):
    """Board points plus pass, for a channels-first (planes, N, N) input."""
    return input_shape[1] * input_shape[2] + 1


'''The dual residual architecture is the strongest
of the architectures tested by DeepMind for AlphaGo
Zero. It consists of an initial convolutional block,
followed by a number (40 for the strongest, 20 as
baseline) of residual blocks. The network is topped
off by two "heads", one to predict policies and one
for value functions.

Policy head: softmax over board points + pass.

legacy_heads=True rebuilds the heads exactly as in models trained before the
TF2 port (heads applied to a channels-first tensor with channels-last
convolutions, sigmoid policy). It exists only so convert_legacy_zero_agent.py
can carry old trained weights over; new models should not use it.
'''
def dual_residual_network(input_shape, blocks=20, legacy_heads=False):
    inputs = Input(shape=input_shape)
    x = to_channels_last()(inputs)
    first_conv = conv_bn_relu_block(name="init")(x)
    res_tower = residual_tower(blocks=blocks)(first_conv)
    if legacy_heads:
        res_tower = Permute((3, 1, 2), name="legacy_to_channels_first")(res_tower)
    policy = policy_head(num_moves(input_shape), legacy=legacy_heads)(res_tower)
    value = value_head(legacy=legacy_heads)(res_tower)
    return Model(inputs=inputs, outputs=[policy, value])


'''The dual convolutional architecture replaces residual
blocks from the dual residual architecture with batch-normalized
convolution layers. The default block size is 12.
'''
def dual_conv_network(input_shape, blocks=12):
    inputs = Input(shape=input_shape)
    x = to_channels_last()(inputs)
    first_conv = conv_bn_relu_block(name="init")(x)
    conv_tower = convolutional_tower(blocks=blocks)(first_conv)
    policy = policy_head(num_moves(input_shape))(conv_tower)
    value = value_head()(conv_tower)
    return Model(inputs=inputs, outputs=[policy, value])


''' In the separate residual architecture policy and value
head don't share a common "tail", i.e. there's two sets of
residual blocks for policy and value networks, respectively.
'''
def separate_residual_network(input_shape, blocks=20):
    inputs_pol = Input(shape=input_shape)
    first_conv_pol = conv_bn_relu_block(name="init")(to_channels_last()(inputs_pol))
    res_tower_pol = residual_tower(blocks=blocks)(first_conv_pol)
    policy = policy_head(num_moves(input_shape))(res_tower_pol)
    policy_model = Model(inputs=inputs_pol, outputs=policy)

    inputs_val = Input(shape=input_shape)
    first_conv_val = conv_bn_relu_block(name="init")(to_channels_last()(inputs_val))
    res_tower_val = residual_tower(blocks=blocks)(first_conv_val)
    value = value_head()(res_tower_val)
    value_model = Model(inputs=inputs_val, outputs=value)

    return policy_model, value_model


'''The separate convolutional network is structurally identical
to the separate residual network, except that residual blocks
are replaced by convolutional blocks.
'''
def separate_conv_network(input_shape, blocks=20):
    inputs_pol = Input(shape=input_shape)
    first_conv_pol = conv_bn_relu_block(name="init")(to_channels_last()(inputs_pol))
    conv_tower_pol =  convolutional_tower(blocks=blocks)(first_conv_pol)
    policy = policy_head(num_moves(input_shape))(conv_tower_pol)
    policy_model = Model(inputs=inputs_pol, outputs=policy)

    inputs_val = Input(shape=input_shape)
    first_conv_val = conv_bn_relu_block(name="init")(to_channels_last()(inputs_val))
    conv_tower_val = convolutional_tower(blocks=blocks)(first_conv_val)
    value = value_head()(conv_tower_val)
    value_model = Model(inputs=inputs_val, outputs=value)

    return policy_model, value_model


def conv_bn_relu_block(name, activation=True, filters=64, kernel_size=(3,3),
                       strides=(1,1), padding="same", init="he_normal"):
    def f(inputs):
        conv = Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding=padding,
                      kernel_initializer=init,
                      name="{}_conv_block".format(name))(inputs)
        batch_norm = BatchNormalization(axis=-1, name="{}_batch_norm".format(name))(conv)
        return Activation("relu", name="{}_relu".format(name))(batch_norm) if activation else batch_norm
    return f


def residual_block(block_num, **args):
    def f(inputs):
        res = conv_bn_relu_block(name="residual_1_{}".format(block_num), activation=True, **args)(inputs)
        res = conv_bn_relu_block(name="residual_2_{}".format(block_num) , activation=False, **args)(res)
        res = add([inputs, res], name="add_{}".format(block_num))
        return Activation("relu", name="{}_relu".format(block_num))(res)
    return f


def residual_tower(blocks, **args):
    def f(inputs):
        x = inputs
        for i in range(blocks):
            x = residual_block(block_num=i)(x)
        return x
    return f

def convolutional_tower(blocks, **args):
    def f(inputs):
        x = inputs
        for i in range(blocks):
            x = conv_bn_relu_block(name=i)(x)
        return x
    return f


def policy_head(units=82, legacy=False):
    def f(inputs):
        conv = Conv2D(filters=2,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      padding="same",
                      name="policy_head_conv_block")(inputs)
        batch_norm = BatchNormalization(axis=1 if legacy else -1,
                                        name="policy_head_batch_norm")(conv)
        activation = Activation("relu", name="policy_head_relu")(batch_norm)
        policy_flat = Flatten()(activation)
        return Dense(units=units, name="policy_head_dense",
                     activation='sigmoid' if legacy else 'softmax')(policy_flat)
    return f


def value_head(legacy=False):
    def f(inputs):
        conv = Conv2D(filters=1,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding="same",
                      name="value_head_conv_block")(inputs)
        batch_norm = BatchNormalization(axis=1 if legacy else -1,
                                        name="value_head_batch_norm")(conv)
        activation = Activation("relu", name="value_head_relu")(batch_norm)
        value_flat = Flatten()(activation)
        dense =  Dense(units= 64, name="value_head_dense", activation="relu")(value_flat)
        return Dense(units= 1, name="value_head_output", activation="tanh")(dense)
    return f
