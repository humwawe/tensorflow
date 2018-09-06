from photoshop.Network import *
#from photoshop.dcganNetwork  import *
from PIL import Image
import numpy as np


def generate():
    g = generator_model()
    g.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1))
    g.load_weights("./model/my_generator_weight.h5")
    random_data = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
    images = g.predict(random_data, verbose=1)

    for i in range(BATCH_SIZE):
        image = images[i] * 127.5 + 127.5
        Image.fromarray(image.astype(np.uint8)).save("generate/image-%s.png" % i)


if __name__ == "__main__":
    generate()
