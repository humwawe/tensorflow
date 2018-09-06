import glob
import numpy as np
from matplotlib import pyplot

from photoshop.Network import *

#bug loss vanish
#from photoshop.dcganNetwork import *

def train():
    data = []
    for image in glob.glob("images/*"):
        image_data = pyplot.imread(image)
        # image_data = image_data.reshape(1,3, 64, 64).astype('float32')
        data.append(image_data)
    input_data = np.array(data)

    input_data = (input_data.astype(np.float32) - 127.5) / 127.5

    g = generator_model()
    d = discriminator_model()
    d_on_g = generator_containing_discriminator(g, d)

    g_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1)
    d_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1)
    g.compile(loss="binary_crossentropy", optimizer=g_optimizer)
    d_on_g.compile(loss="binary_crossentropy", optimizer=g_optimizer)
    d.trainable = True
    d.compile(loss="binary_crossentropy", optimizer=d_optimizer)

    for epoch in range(EPOCHS):
        for index in range(int(input_data.shape[0] / BATCH_SIZE)):
            input_batch = input_data[index * BATCH_SIZE: (index + 1) * BATCH_SIZE]
            random_data = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))  # 随机数据生成图片
            generated_images = g.predict(random_data, verbose=0)
            input_batch = np.concatenate((input_batch, generated_images))
            output_batch = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = d.train_on_batch(input_batch, output_batch)
            d.trainable = False
            random_data = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            g_loss = d_on_g.train_on_batch(random_data, [1] * BATCH_SIZE)
            d.trainable = True

            print("Epoch {}, 第 {} 步, 生成器的损失: {:.3f}, 判别器的损失: {:.3f}".format(epoch, index, g_loss, d_loss))
            g.save_weights("./model/my_generator_weight.h5", True)

        if epoch % 10 == 9:
            g.save_weights("./model/my_generator_weight.h5", True)


if __name__ == "__main__":
    train()
