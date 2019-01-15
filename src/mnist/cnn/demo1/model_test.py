import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    mnist = input_data.read_data_sets("../../mnist_data/", one_hot=True)
    test_x = mnist.test.images[:3000]
    test_y = mnist.test.labels[:3000]

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./model/mnist_model.meta')
        saver.restore(sess, './model/mnist_model')
        graph = tf.get_default_graph()

        input = graph.get_operation_by_name('input_x').outputs[0]
        result = tf.get_collection('result')[0]

        test_output = sess.run(result, feed_dict={input: test_x})
        inferenced_y = np.argmax(test_output, 1)
        print(inferenced_y[:30], 'Inferenced numbers')
        print(np.argmax(test_y[:30], 1), 'Real numbers')
        print(accuracy_score(np.argmax(test_y, 1), inferenced_y))
