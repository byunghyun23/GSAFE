import tensorflow as tf
import keras.backend as K


def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[1], -1, x.shape[-1]])


def self_attention(x, channels):
    f = tf.keras.layers.Conv2D(filters=channels, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    g = tf.keras.layers.Conv2D(filters=channels, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    h = tf.keras.layers.Conv2D(filters=channels, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)

    s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]
    beta = tf.nn.sigmoid(s)  # attention map
    # beta = tf.nn.softmax(s)  # attention map

    o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
    # gamma = get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

    # o = tf.reshape(o, shape=x.shape[1:])  # [bs, h, w, C]
    o = tf.reshape(o, shape=tf.shape(x))  # [bs, h, w, C]
    o = tf.keras.layers.Conv2D(name='attention_output_' + str(channels), filters=channels, kernel_size=(1, 1), strides=(1, 1), padding='same')(o)

    # x = gamma * o + x
    x = o + x

    return x


def GSAFE():
    leaky_relu = tf.nn.leaky_relu

    input = tf.keras.layers.Input(shape=(256, 256, 3))

    x0 = tf.keras.layers.Conv2D(dilation_rate=(2, 2), filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same',
                                activation=leaky_relu)(input)
    x1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(9, 9), strides=(1, 1), padding='same', activation='relu')(x0)
    # a1 = tf.keras.layers.Permute((1, 3, 2))(x1)
    # a1 = tf.keras.layers.Dense(x1.shape[2], activation='sigmoid')(a1)
    # a1 = tf.keras.layers.Permute((1, 3, 2))(a1)
    # m1 = tf.keras.layers.Multiply()([x1, a1])
    # r1 = tf.keras.layers.Add(name='Add1')([x1, m1])
    x1 = self_attention(x1, channels=16)
    x1 = tf.keras.layers.Add(name='attention_output_add_16')([x0, x1])

    x2 = tf.keras.layers.Conv2D(dilation_rate=(2, 2), filters=32, kernel_size=(4, 4), strides=(1, 1), padding='same',
                                activation=leaky_relu)(x1)
    x3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(x2)
    # a2 = tf.keras.layers.Permute((1, 3, 2))(x3)
    # a2 = tf.keras.layers.Dense(x3.shape[2], activation='sigmoid')(a2)
    # a2 = tf.keras.layers.Permute((1, 3, 2))(a2)
    # m2 = tf.keras.layers.Multiply()([x3, a2])l
    # r2 = tf.keras.layers.Add(name='Add2')([x3, m2])
    x3 = self_attention(x3, channels=32)
    x3 = tf.keras.layers.Add(name='attention_output_add_32')([x2, x3])

    x4 = tf.keras.layers.Conv2D(dilation_rate=(2, 2), filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                activation=leaky_relu)(x3)
    x5 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x4)
    # a3 = tf.keras.layers.Permute((1, 3, 2))(x5)
    # a3 = tf.keras.layers.Dense(x5.shape[2], activation='sigmoid')(a3)
    # a3 = tf.keras.layers.Permute((1, 3, 2))(a3)
    # m3 = tf.keras.layers.Multiply()([x5, a3])
    # r3 = tf.keras.layers.Add(name='Add3')([x5, m3])
    x5 = self_attention(x5, channels=64)
    x5 = tf.keras.layers.Add(name='attention_output_add_64')([x4, x5])

    x6 = tf.keras.layers.Conv2D(dilation_rate=(2, 2), filters=128, kernel_size=(2, 2), strides=(1, 1), padding='same',
                                activation=leaky_relu)(x5)
    x7 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x6)
    # a4 = tf.keras.layers.Permute((1, 3, 2))(x7)
    # a4 = tf.keras.layers.Dense(x7.shape[2], activation='sigmoid')(a4)
    # a4 = tf.keras.layers.Permute((1, 3, 2))(a4)
    # m4 = tf.keras.layers.Multiply()([x7, a4])
    # r4 = tf.keras.layers.Add(name='Add4')([x7, m4])
    x7 = self_attention(x7, channels=128)
    x7 = tf.keras.layers.Add(name='attention_output_add_128')([x6, x7])

    output = tf.keras.layers.Conv2D(filters=3, kernel_size=(5, 5), strides=(1, 1), padding='same')(x7)
    output = tf.keras.layers.Add(name='add_output')([input, output])

    model = tf.keras.models.Model(inputs=input, outputs=output)

    return model

