
import tensorflow as tf

if __name__ == '__main__':
    hw = tf.constant("Hello World")
    with tf.Session() as sess:
        print(sess.run(hw))
