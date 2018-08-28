import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == '__main__':

    points_num = 100
    vectors = []
    #  y = 0.1 * x + 0.2
    for i in range(points_num):
        x1 = np.random.normal(0.0, 0.66)
        y1 = 0.1 * x1 + 0.2 + np.random.normal(0.0, 0.04)
        vectors.append([x1, y1])

    x_data = [v[0] for v in vectors]
    y_data = [v[1] for v in vectors]

    # 图像 1 ：展示 100 个随机数据点
    plt.plot(x_data, y_data, 'r*', label="Original data")
    plt.title("Linear Regression using Gradient Descent")
    plt.legend()
    plt.show()

    # 构建线性回归模型
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    y = W * x_data + b

    # ((y - y_data) ^ 2) 之和 / N
    loss = tf.reduce_mean(tf.square(y - y_data))

    optimizer = tf.train.GradientDescentOptimizer(0.5)  # 学习率为 0.5
    train = optimizer.minimize(loss)

    sess = tf.Session()

    # 初始化数据流图中的所有变量
    init = tf.global_variables_initializer()
    sess.run(init)

    # 训练 20 步
    for step in range(20):
        # 优化每一步
        sess.run(train)
        print("Step=%d, Loss=%f, [Weight=%f Bias=%f]" % (step, sess.run(loss), sess.run(W), sess.run(b)))

    # 图像 2 ：绘制所有的点并且绘制出最佳拟合的直线
    plt.plot(x_data, y_data, 'r*', label="Original data")
    plt.title("Linear Regression using Gradient Descent")
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label="Fitted line")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    sess.close()
