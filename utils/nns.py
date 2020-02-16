import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, UpSampling2D, Conv2D, add, AveragePooling2D, LeakyReLU, Input, Reshape, Flatten
import math


def AdaIN(x):
    '''
    :param x: x[0] = Image representation. Shape: (batch size, height, width, channels)
              x[1] = Scaling parameter for style
              x[2] = Bias parameter for style
    :return:
    '''
    mean = tf.reduce_mean(x[0], axis=[1, 2], keepdims=True)
    std = tf.math.reduce_std(x[0], axis=[1, 2], keepdims=True) + 1e-7
    y = (x[0] - mean) / std

    pool_shape = [-1, 1, 1, y.shape[-1]]
    y_s = tf.reshape(x[1], pool_shape)
    y_b = tf.reshape(x[2], pool_shape)

    return y * y_s + y_b


def crop_to_fit(x):
    '''
    We want to fit the noise to the specific layer it is being fed
    :param x:  x[0] = noise, x[1] = input tensor
    :return:
    '''
    height, width = x[1].shape[1], x[1].shape[2]
    return x[0][:, :height * 2, :width * 2, :]


def g_block(input_tensor, latent_vector, noise, filters):
    '''
    Block for the synthesis network
    :param input_tensor:
    :param latent_vector: In the paper, w.
    :param noise: Noise vector without cropping
    :param filters:
    :return:
    '''
    y_s = Dense(filters, bias_initializer='ones')(latent_vector)
    y_b = Dense(filters)(latent_vector)

    noise = Lambda(crop_to_fit)([noise, input_tensor])
    noise = Dense(filters)(noise)

    out = UpSampling2D()(input_tensor)
    out = Conv2D(filters, 3, padding='same')(out)
    out = add([out, noise])
    out = Lambda(AdaIN)([out, y_s, y_b])
    out = LeakyReLU(0.2)(out)

    return out

def d_block(input_tensor, filters):
    out = Conv2D(filters, 3, padding='same')(input_tensor)
    out = LeakyReLU(0.2)(out)
    out = AveragePooling2D()(out)
    return out

def generator_network(input_dim, mapping_size, noise_shape, out_dim):
    out_size = math.log2(out_dim)
    assert int(out_size) == out_size, 'Output dimension must be 2**n'
    noise_input = Input(noise_shape)
    latent_input = Input([input_dim])
    # Mapping network
    latent = Dense(input_dim)(latent_input)
    for layer in range(mapping_size - 1):
        latent = Dense(input_dim)(latent)
        latent = LeakyReLU(0.2)(latent)

    x = Dense(1)(latent_input)
    x = Lambda(lambda y: y*0 + 1)(x)

    # Reshape the latent network to a small image (4x4x64)
    x = Dense(4 * 4 * out_dim)(x)
    x = LeakyReLU(0.2)(x)
    x = Reshape([4, 4, out_dim])(x)

    for layer in range(int(out_size), 2, -1):
        x = g_block(x, latent, noise_input, 2 ** layer)

    image_output = Conv2D(3, 1, padding='same', activation='sigmoid')(x)
    return tf.keras.Model(inputs=[latent_input, noise_input], outputs=image_output)

def discriminator_network(input_shape):
    assert input_shape[0] == input_shape[1], 'Image must be a square.'
    input_dim = input_shape[0]
    channels = input_shape[2]
    n_layers = int(math.log2(input_dim))

    input_image = Input(input_shape) # 256x256x3

    for layer in range(4, n_layers + 1):
        if layer == 4:
            x = d_block(input_image, layer)
        else:
            x = d_block(x, layer)

    x = Conv2D(input_dim, channels, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)

    class_output = Dense(1)(x)
    return tf.keras.Model(inputs=input_image, outputs=class_output)


if __name__ == '__main__':
    INPUT_DIM = 128
    MAPPING_SIZE = 8
    NOISE_SHAPE = (128, 128, 1)
    OUT_DIM = 128

    gen = generator_network(INPUT_DIM, MAPPING_SIZE, NOISE_SHAPE, OUT_DIM)
    disc = discriminator_network((128, 128, 3))
