import tensorflow as tf
import numpy as np

EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
BETA_1 = 0.5

def discriminator_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(64, 64, 3)))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.5))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.5))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.5))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.5))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation("sigmoid"))
    return model


def generator_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(input_dim=100, units=4 * 4 * 512))
    model.add(tf.keras.layers.Reshape((4, 4, 512), input_shape=(4 * 4 * 512,)))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.5))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(5, 5), strides=(2, 2), padding="same"))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.5))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(5, 5), strides=(2, 2), padding="same"))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.5))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding="same"))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.5))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(5, 5), strides=(2, 2), padding="same"))
    model.add(tf.keras.layers.Activation("tanh"))
    return model

def generator_containing_discriminator(generator, discriminator):
    model = tf.keras.models.Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model


if __name__ == '__main__':
    a = np.array([[1, 2, 2, 2]], dtype=np.float32)
    # a = np.reshape(a, [1,2, 2, 1]) 要求输入是4维(samples,col,row,channel)
    x = tf.constant(a, dtype=tf.float32)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Reshape((2, 2, 1), input_shape=(4,)))
    model.add(tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(2, 2), strides=(2, 2),
                                              kernel_initializer=tf.ones_initializer()))

    # Step1  补零。在每个元素后插 strides-1 个 0，之后将原shape扩大strides倍
    # Step2  卷积。按kernel_size做步长为1的卷积，如果为same，大小原step1的大小一致，因此可能需要继续在左右两边补零
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        xxx = model.predict(x, steps=1)
        print(xxx)
