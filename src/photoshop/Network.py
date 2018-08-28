import tensorflow as tf

EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
BETA_1 = 0.5


def discriminator_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), padding='same', input_shape=(64, 64, 3)))
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation("sigmoid"))
    return model


def generator_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(input_dim=100, units=1024))
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.Dense(128 * 8 * 8))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.Reshape((8, 8, 128), input_shape=(128 * 8 * 8,)))
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (5, 5), padding="same"))
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (5, 5), padding="same"))
    model.add(tf.keras.layers.Activation("tanh"))
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(3, (5, 5), padding="same"))
    model.add(tf.keras.layers.Activation("tanh"))
    return model


def generator_containing_discriminator(generator, discriminator):
    model = tf.keras.models.Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model
