# # PFLD: A Practical Facial Landmark Detector

# import sys
# import time

# from keras.models import Model
# from keras.layers import *
# from keras import backend as K
# from keras.utils.vis_utils import plot_model
# from keras.utils import vis_utils


# def _conv_block(inputs, filters, kernel, strides, dilation_rate=1, padding='same'):
#     """Convolution Block
#     This function defines a 2D convolution operation with BN and relu6.
#     # Arguments
#         inputs: Tensor, input tensor of conv layer.
#         filters: Integer, the dimensionality of the output space.
#         kernel: An integer or tuple/list of 2 integers, specifying the
#             width and height of the 2D convolution window.
#         strides: An integer or tuple/list of 2 integers,
#             specifying the strides of the convolution along the width and height.
#             Can be a single integer to specify the same value for
#             all spatial dimensions.
#     # Returns
#         Output tensor.
#     """
#     channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
#     x = Conv2D(filters, kernel, padding=padding, strides=strides,
#                dilation_rate=dilation_rate)(inputs)
#     x = BatchNormalization(axis=channel_axis)(x)

#     return Activation('relu')(x)


# def _depthwise_block(inputs, kernel, strides, padding='same'):
#     '''Depthwise separable 2D convolution block'''

#     assert isinstance(kernel, (tuple, int))
#     assert isinstance(strides, (tuple, int))

#     channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
#     x = DepthwiseConv2D(kernel_size=kernel, strides=strides,
#                         depth_multiplier=1, padding=padding)(inputs)
#     x = BatchNormalization(axis=channel_axis)(x)

#     return Activation('relu')(x)


# def _bottleneck(inputs, filters, kernel, t, s, alpha, r=False):
#     """Bottleneck
#     This function defines a basic bottleneck structure.
#     # Arguments
#         inputs: Tensor, input tensor of conv layer.
#         filters: Integer, the dimensionality of the output space.
#         kernel: An integer or tuple/list of 2 integers, specifying the
#             width and height of the 2D convolution window.
#         t: Integer, expansion factor.
#             t is always applied to the input size.
#         s: An integer or tuple/list of 2 integers,specifying the strides
#             of the convolution along the width and height.Can be a single
#             integer to specify the same value for all spatial dimensions.
#         r: Boolean, Whether to use the residuals.
#     # Returns
#         Output tensor.
#     """

#     channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
#     tchannel = K.int_shape(inputs)[channel_axis] * t
#     filters = _make_divisible(filters * alpha)

#     x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

#     x = DepthwiseConv2D(kernel, strides=(
#         s, s), depth_multiplier=1, padding='same')(x)
#     x = BatchNormalization(axis=channel_axis)(x)
#     x = ReLU(max_value=6)(x)

#     x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
#     x = BatchNormalization(axis=channel_axis)(x)

#     if r:
#         x = add([x, inputs])
#     return x


# def _inverted_residual_block(inputs, filters, kernel, t, strides, n, alpha=1):
#     """Inverted Residual Block
#     This function defines a sequence of 1 or more identical layers.
#     # Arguments
#         inputs: Tensor, input tensor of conv layer.
#         filters: Integer, the dimensionality of the output space.
#         kernel: An integer or tuple/list of 2 integers, specifying the
#             width and height of the 2D convolution window.
#         t: Integer, expansion factor.
#             t is always applied to the input size.
#         s: An integer or tuple/list of 2 integers,specifying the strides
#             of the convolution along the width and height.Can be a single
#             integer to specify the same value for all spatial dimensions.
#         n: Integer, layer repeat times.
#     # Returns
#         Output tensor.
#     """
#     x = _bottleneck(inputs, filters, kernel, t, strides, alpha=alpha)

#     for i in range(1, n):
#         x = _bottleneck(x, filters, kernel, t, 1, alpha=alpha, r=True)

#     return x

# # https://github.com/titu1994/MobileNetworks/blob/master/mobilenets.py


# def _make_divisible(v, divisor=8, min_value=8):
#     if min_value is None:
#         min_value = divisor

#     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#     # Make sure that round down does not go down by more than 10%.
#     if new_v < 0.9 * v:
#         new_v += divisor
#     return new_v


# def PFLDNetBackbone(input_shape, output_nodes, alpha=1):
#     """
#     This function defines a PFLDNet architectures.
#     # Arguments
#         input_shape: An integer or tuple/list of 3 integers, shape
#             of input tensor.
#         output_nodes: Integer, number of classes.
#         alpha: width parameter.
#     # Returns
#         PFLDNet model.
#     """

#     inputs = Input(shape=input_shape)
#     # https://mp.weixin.qq.com/s/0oMqwQn2UlYYk557sbPBsQ
#     x = ZeroPadding2D(padding=(1, 1))(inputs)
#     x = _conv_block(x, 64, (3, 3), strides=1, dilation_rate=2)
#     x = _depthwise_block(x, (3, 3), strides=2)
#     s1_b = _inverted_residual_block(
#         x, 64, (3, 3), t=2, strides=2, n=5, alpha=alpha)
#     x = _inverted_residual_block(
#         s1_b, 128, (3, 3), t=2, strides=2, n=1, alpha=alpha)
#     x = _inverted_residual_block(
#         x, 128, (3, 3), t=4, strides=1, n=6, alpha=alpha)
#     s1 = _inverted_residual_block(
#         x, 256, (3, 3), t=2, strides=1, n=1, alpha=alpha)
#     s2 = _conv_block(s1, 256, (3, 3), strides=1, dilation_rate=2)
#     s3 = _conv_block(s2, 256, (3, 3), strides=1, dilation_rate=2)

#     # 106 Landmarks branch
#     # t1_g = Flatten()(s1)
#     # t2_g = Flatten()(s2)
#     # t3_g = Flatten()(s3)
#     # t1_212 = Dense(units=output_nodes, name='b1_s1')(t1_g)
#     # t2_212 = Dense(units=output_nodes, name='b1_s2')(t1_g)
#     # t3_212 = Dense(units=output_nodes, name='b1_s3')(t1_g)
#     # t1_out = Add(name='b1_s')([t1_212, t2_212, t3_212])
#     t1_g = GlobalAveragePooling2D()(s1)
#     t2_g = GlobalAveragePooling2D()(s2)
#     t3_g = GlobalAveragePooling2D()(s3)
#     concat = Concatenate()([t1_g, t2_g, t3_g])
#     t1_out = Dense(units=output_nodes, name='b1_s')(concat)

#     # Pose branch
#     v1 = _conv_block(s1_b, 128, (3, 3), strides=2)
#     v2 = _conv_block(v1, 128, (3, 3), strides=1)
#     v3 = _conv_block(v2, 32, (3, 3), strides=2)
#     v4 = _conv_block(v3, 128, (7, 7), strides=1, padding='valid')
#     t2_out = Dense(units=3, name='b2_s')(Flatten()(v4))

#     # TODO angle...

#     # Merge branch
#     model = Model(inputs, [t1_out, t2_out])

#     return model


# if __name__ == '__main__':

#     # Testing designed network
#     model = PFLDNetBackbone((112, 112, 3), 212, alpha=1.0)
#     vis = True

#     if vis:
#         model.summary()
#         # plot_model(model, to_file='PFLDNet.png', show_shapes=True)

#     # inputs = np.random.randn(1, 112, 112, 3)

#     # for i in range(100):
#     #     start = time.time()
#     #     model.predict(inputs, batch_size=1)
#     #     print("[info] time use {}".format(time.time() - start))


"""MobileNet v3 small models for Keras.
# Reference
    [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244?context=cs)
"""
from keras import backend as K
from keras.layers import *
from keras.models import Model
from keras.utils.generic_utils import get_custom_objects


def hard_swish(x):
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0


def relu6(x):
    return K.relu(x, max_value=6.0)

# Custom activation function
get_custom_objects().update({'hard_swish': Activation(hard_swish)})
get_custom_objects().update({'relu6': Activation(relu6)})


class MobileNetBase:
    def __init__(self, shape, n_class):
        self.shape = shape
        self.n_class = n_class

    # def _relu6(self, x):
    #     """Relu 6
    #     """
    #     return K.relu(x, max_value=6.0)

    # def _hard_swish(self, x):
    #     """Hard swish
    #     """
    #     return x * K.relu(x + 3.0, max_value=6.0) / 6.0

    def _return_activation(self, x, nl):
        """Convolution Block
        This function defines a activation choice.

        # Arguments
            x: Tensor, input tensor of conv layer.
            nl: String, nonlinearity activation type.

        # Returns
            Output tensor.
        """
        if nl == 'HS':
            x = Activation(hard_swish)(x)
        if nl == 'RE':
            x = Activation(relu6)(x)

        return x

    def _conv_block(self, inputs, filters, kernel, strides, nl):
        """Convolution Block
        This function defines a 2D convolution operation with BN and activation.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.
            nl: String, nonlinearity activation type.

        # Returns
            Output tensor.
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
        x = BatchNormalization(axis=channel_axis)(x)

        return self._return_activation(x, nl)

    def _squeeze(self, inputs):
        """Squeeze and Excitation.
        This function defines a squeeze structure.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
        """
        # input_channels = int(inputs.shape[-1])
        input_channels = inputs._keras_shape[-1]

        x = GlobalAveragePooling2D()(inputs)
        x = Dense(int(input_channels/4), activation='relu')(x)
        x = Dense(input_channels, activation='hard_sigmoid')(x)
        x = Reshape((1, 1, -1))(x)
        x = multiply([inputs, x])

        return x

    def _bottleneck(self, inputs, filters, kernel, e, s, squeeze, nl):
        """Bottleneck
        This function defines a basic bottleneck structure.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            e: Integer, expansion factor.
                t is always applied to the input size.
            s: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.
            squeeze: Boolean, Whether to use the squeeze.
            nl: String, nonlinearity activation type.

        # Returns
            Output tensor.
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        input_shape = K.int_shape(inputs)
        tchannel = e
        r = s == 1 and input_shape[3] == filters
        x = self._conv_block(inputs, tchannel, (1, 1), (1, 1), nl)

        x = DepthwiseConv2D(kernel, strides=(
            s, s), depth_multiplier=1, padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)

        if squeeze:
            # x = Lambda(lambda x: x * self._squeeze(x))(x)
            x = self._squeeze(x)

        x = self._return_activation(x, nl)

        x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)

        if r:
            x = Add()([x, inputs])

        return x

    def build(self):
        raise NotImplementedError


class MobileNetV3(MobileNetBase):
    def __init__(self, shape, n_class):
        """Init.

        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            n_class: Integer, number of classes.

        # Returns
            MobileNetv3 model.
        """
        super(MobileNetV3, self).__init__(shape, n_class)

    def build(self):
        """build MobileNetV3 Small.

        # Arguments
            plot: Boolean, weather to plot model.

        # Returns
            model: Model, model.
        """
        inputs = Input(shape=self.shape)

        x = self._conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='HS')

        # x = self._bottleneck(x, 16, (3, 3), e=16, s=2, squeeze=True, nl='RE')
        # x = self._bottleneck(x, 24, (3, 3), e=72, s=2, squeeze=False, nl='RE')
        # x = self._bottleneck(x, 24, (3, 3), e=88, s=1, squeeze=False, nl='RE')
        # x = self._bottleneck(x, 40, (5, 5), e=96, s=2, squeeze=True, nl='HS')
        # x = self._bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS')
        # x = self._bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS')
        # x = self._bottleneck(x, 48, (5, 5), e=120, s=1, squeeze=True, nl='HS')
        # x = self._bottleneck(x, 48, (5, 5), e=144, s=1, squeeze=True, nl='HS')
        # x = self._bottleneck(x, 96, (5, 5), e=288, s=2, squeeze=True, nl='HS')
        # x = self._bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS')
        # x = self._bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 16, (3, 3), e=16, s=1, squeeze=False, nl='RE')
        x = self._bottleneck(x, 24, (3, 3), e=64, s=2, squeeze=False, nl='RE')
        x = self._bottleneck(x, 24, (3, 3), e=72, s=1, squeeze=False, nl='RE')
        x = self._bottleneck(x, 40, (5, 5), e=72, s=2, squeeze=True, nl='RE')
        x = self._bottleneck(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE')
        x = self._bottleneck(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE')
        x = self._bottleneck(x, 80, (3, 3), e=240, s=2, squeeze=False, nl='HS')
        x = self._bottleneck(x, 80, (3, 3), e=200, s=1, squeeze=False, nl='HS')
        x = self._bottleneck(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='HS')
        x = self._bottleneck(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='HS')
        x = self._bottleneck(x, 112, (3, 3), e=480, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 112, (3, 3), e=672, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 160, (5, 5), e=672, s=2, squeeze=True, nl='HS')
        x = self._bottleneck(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='HS')

        # x = self._conv_block(x, 576, (1, 1), strides=(1, 1), nl='HS')
        # x = GlobalAveragePooling2D()(x)
        # x = Reshape((1, 1, 576))(x)
        x = self._conv_block(x, 960, (1, 1), strides=(1, 1), nl='HS')
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 960))(x)

        x = Conv2D(1280, (1, 1), padding='same')(x)
        t1_0 = self._return_activation(x, 'HS')
        t1_1 = Conv2D(self.n_class, (1, 1), padding='same')(t1_0)
        t1_out = Reshape((self.n_class,), name='b1_s')(t1_1)

        t2 = Conv2D(3, (1, 1), padding='same')(t1_0)
        t2_out = Reshape((3,), name='b2_s')(t2)
        # Merge branch
        model = Model(inputs, [t1_out, t2_out])

        return model


if __name__ == "__main__":

    model = MobileNetV3((112, 112, 3), 202).build()
    model.summary()
