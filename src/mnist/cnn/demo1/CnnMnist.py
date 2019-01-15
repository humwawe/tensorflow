import numpy as np
import tensorflow as tf

# 下载并载入 MNIST 手写数字库（55000 * 28 * 28）55000 张训练图像
from tensorflow.examples.tutorials.mnist import input_data


def network(input_x_images):
    conv1 = tf.layers.conv2d(
        inputs=input_x_images,  # 形状 [28, 28, 1]
        filters=32,  # 32 个过滤器，输出的深度（depth）是32
        kernel_size=[5, 5],  # 过滤器在二维的大小是(5 * 5)
        strides=1,  # 步长是1
        padding='same',  # same 表示输出的大小不变，因此需要在外围补零 2 圈
        activation=tf.nn.relu  # 激活函数是 Relu
    )  # 形状 [28, 28, 32]

    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,  # 形状 [28, 28, 32]
        pool_size=[2, 2],  # 过滤器在二维的大小是（2 * 2）
        strides=2  # 步长是 2
    )  # 形状 [14, 14, 32]

    conv2 = tf.layers.conv2d(
        inputs=pool1,  # 形状 [14, 14, 32]
        filters=64,
        kernel_size=[5, 5],
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )  # 形状 [14, 14, 64]

    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,  # 形状 [14, 14, 64]
        pool_size=[2, 2],
        strides=2
    )  # 形状 [7, 7, 64]

    flat = tf.reshape(pool2, [-1, 7 * 7 * 64])  # 形状 [7 * 7 * 64, ]
    dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.5)
    logits = tf.layers.dense(inputs=dropout, units=10)

    return logits


if __name__ == '__main__':
    mnist = input_data.read_data_sets("../../mnist_data/", one_hot=True)
    # None 表示张量（Tensor）的第一个维度可以是任何长度
    input_x = tf.placeholder(tf.float32, [None, 28 * 28], name='input_x')
    output_y = tf.placeholder(tf.int32, [None, 10])
    input_x_images = tf.reshape(input_x, [-1, 28, 28, 1])  # 改变形状之后的输入
    test_x = mnist.test.images[:3000]
    test_y = mnist.test.labels[:3000]

    logits = network(input_x_images)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y, logits=logits)
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    tf.add_to_collection('result', logits)

    accuracy = tf.metrics.accuracy(
        labels=tf.argmax(output_y, axis=1),
        predictions=tf.argmax(logits, axis=1), )[1]

    saver = tf.train.Saver()
    sess = tf.Session()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    for i in range(500):
        batch = mnist.train.next_batch(50)
        train_loss, train_op_ = sess.run([loss, train_op], {input_x: batch[0], output_y: batch[1]})
        if i % 100 == 0:
            test_accuracy = sess.run(accuracy, {input_x: test_x, output_y: test_y})
            print("Step=%d, Train loss=%.4f, [Test accuracy=%.2f]" % (i, train_loss, test_accuracy))
    saver.save(sess, './model/mnist_model')

    test_output = sess.run(logits, {input_x: test_x[:20]})
    inferenced_y = np.argmax(test_output, 1)
    print(inferenced_y, 'Inferenced numbers')
    print(np.argmax(test_y[:20], 1), 'Real numbers')
