import tensorflow as tf
from tensorflow.keras.layers import Input, SeparableConv2D, PReLU, Multiply


class SingleBlock(tf.keras.layers.Layer):
    def __init__(self, dim=256, out_shape=512, *args, **kwargs):
        super(SingleBlock, self).__init__()
        self.dim = dim
        self.out_shape = out_shape
        self.l1 = SeparableConv2D(
            self.dim,
            5,
            1,
            "same",
            kernel_initializer="he_uniform",
            depthwise_initializer="he_uniform",
        )
        self.l2 = PReLU()
        self.l3 = SeparableConv2D(
            self.out_shape,
            5,
            1,
            "same",
            kernel_initializer="he_uniform",
            depthwise_initializer="he_uniform",
        )

    def call(self, inputs, **kwargs):
        x = self.l1(inputs)
        x = self.l2(x)
        x = self.l3(x)
        return x

    def get_config(self):
        config = super(SingleBlock, self).get_config()
        config.update({"dim": self.dim, "out_shape": self.out_shape})
        return config


def create_model(input_shape=(2, 400, 512), num_blocks=1):
    inp = Input(input_shape)
    block_list = []
    if num_blocks == 1:
        b1 = SingleBlock()(inp)
        return tf.keras.models.Model(inp, [b1, b1])

    elif num_blocks % 2 == 0:
        for i in range(num_blocks // 2):
            if i == 0:
                b = SingleBlock()(inp)
            else:
                b = SingleBlock()(b)
            block_list.append(b)
        for i in reversed(block_list):
            b = SingleBlock()(b)
            b = Multiply()([b, i])
        b = SeparableConv2D(
            input_shape[-1],
            5,
            1,
            "same",
            kernel_initializer="he_uniform",
            depthwise_initializer="he_uniform",
        )(b)
        return tf.keras.models.Model(inp, [b, b])

    else:
        for i in range(num_blocks // 2):
            if i == 0:
                b = SingleBlock()(inp)
            else:
                b = SingleBlock()(b)
            block_list.append(b)
        b = SingleBlock()(b)
        for i in reversed(block_list):
            b = SingleBlock()(b)
            b = Multiply()([b, i])
        b = SeparableConv2D(
            input_shape[-1],
            5,
            1,
            "same",
            kernel_initializer="he_uniform",
            depthwise_initializer="he_uniform",
        )(b)
        return tf.keras.models.Model(inp, [b, b])
