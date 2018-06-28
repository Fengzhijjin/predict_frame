# coding:utf-8
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import read as rd
import os
from PIL import Image
import shutil

learning_rate = 0.01
max_samples = 400000
display_step = 100


def conv2d_bn(input_data, kernel_num, filter_size=[2, 2], stride=1,
              padding='SAME', is_training=True, activation_fn=tf.nn.relu,
              scope=None):
    res = slim.conv2d(
        input_data,
        kernel_num,
        filter_size,
        stride=stride,
        padding=padding,
        activation_fn=activation_fn,
        scope=scope)
    res = slim.batch_norm(res, decay=0.01, is_training=is_training,
                          scope=scope)
    return res


def conv2d_tr(input_data, filter_shape, output_shape, strides,
              padding='SAME', is_training=True, scope=None):
    filter_in = tf.Variable(tf.random_normal(filter_shape))
    res = tf.nn.conv2d_transpose(input_data, filter_in, output_shape, strides,
                                 padding)
    res = slim.batch_norm(res, decay=0.01, is_training=is_training,
                          scope=scope)
    return res


def Lower_sample(x, is_training=True):
    with tf.variable_scope('Lower_sample/net1') as scope:
        net = conv2d_bn(x, 16, filter_size=[3, 3], stride=[1, 1],
                        is_training=is_training, scope=scope)
        # shape = (none, 180, 240, 16)
    with tf.variable_scope('Lower_sample/net2') as scope:
        net = conv2d_bn(net, 64, filter_size=[3, 3], stride=[2, 2],
                        is_training=is_training, scope=scope)
        # shape = (none, 90, 120, 64)
    with tf.variable_scope('Lower_sample/net3') as scope:
        net = conv2d_bn(net, 128, filter_size=[3, 3], stride=[1, 1],
                        is_training=is_training, scope=scope)
        # shape = (none, 90, 120, 128)
    with tf.variable_scope('Lower_sample/net4') as scope:
        net = conv2d_bn(net, 128, filter_size=[1, 1], stride=[1, 1],
                        is_training=is_training, scope=scope)
        # shape = (none, 90, 120, 128)
    with tf.variable_scope('Lower_sample/net5') as scope:
        net = conv2d_bn(net, 256, filter_size=[3, 3], stride=[2, 2],
                        is_training=is_training, scope=scope)
        # shape = (none, 45, 60, 256)
    with tf.variable_scope('Lower_sample/net6') as scope:
        net = conv2d_bn(net, 256, filter_size=[3, 3], stride=[1, 1],
                        is_training=is_training, scope=scope)
        # shape = (none, 45, 60, 256)
    with tf.variable_scope('Lower_sample/net7') as scope:
        net = conv2d_bn(net, 256, filter_size=[1, 1], stride=[1, 1],
                        is_training=is_training, scope=scope)
        # shape = (none, 45, 60, 256)
    with tf.variable_scope('Lower_sample/net8') as scope:
        net = conv2d_bn(net, 512, filter_size=[3, 3], stride=[2, 2],
                        is_training=is_training, scope=scope)
        # shape = (none, 23, 30, 512)
    with tf.variable_scope('Lower_sample/net9') as scope:
        net = conv2d_bn(net, 256, filter_size=[1, 1], stride=[1, 1],
                        is_training=is_training, scope=scope)
        # shape = (none, 23, 30, 256)

    net = tf.transpose(net, [3, 0, 1, 2])

    return net


def Upper_sample(x, is_training=True):
    x = tf.reshape(x, [-1, 16, 23, 30])
    x = tf.transpose(x, [1, 2, 3, 0])
    # shape = (none, 23, 30, 256)

    net = conv2d_tr(x, [1, 1, 512, 256], [16, 23, 30, 512], [1, 1, 1, 1],
                    is_training=is_training, scope='Upper_sample/net9')
    net = tf.nn.relu(net)
    # shape = (none, 23, 30, 512)

    net = conv2d_tr(net, [3, 3, 256, 512], [16, 45, 60, 256], [1, 2, 2, 1],
                    is_training=is_training, scope='Upper_sample/net8')
    net = tf.nn.relu(net)
    # shape = (none, 45, 60, 256)

    net = conv2d_tr(net, [1, 1, 256, 256], [16, 45, 60, 256], [1, 1, 1, 1],
                    is_training=is_training, scope='Upper_sample/net7')
    net = tf.nn.relu(net)
    # shape = (none, 45, 60, 256)

    net = conv2d_tr(net, [3, 3, 256, 256], [16, 45, 60, 256], [1, 1, 1, 1],
                    is_training=is_training, scope='Upper_sample/net6')
    net = tf.nn.relu(net)
    # shape = (none, 45, 60, 128)

    net = conv2d_tr(net, [3, 3, 128, 256], [16, 90, 120, 128], [1, 2, 2, 1],
                    is_training=is_training, scope='Upper_sample/net5')
    net = tf.nn.relu(net)
    # shape = (none, 90, 120, 128)

    net = conv2d_tr(net, [1, 1, 128, 128], [16, 90, 120, 128], [1, 1, 1, 1],
                    is_training=is_training, scope='Upper_sample/net4')
    net = tf.nn.relu(net)
    # shape = (none, 90, 120, 128)

    net = conv2d_tr(net, [3, 3, 64, 128], [16, 90, 120, 64], [1, 1, 1, 1],
                    is_training=is_training, scope='Upper_sample/net3')
    net = tf.nn.relu(net)
    # shape = (none, 90, 120, 64)

    net = conv2d_tr(net, [3, 3, 16, 64], [16, 180, 240, 16], [1, 2, 2, 1],
                    is_training=is_training, scope='Upper_sample/net2')
    net = tf.nn.relu(net)
    # shape = (none, 180, 240, 16)

    net = conv2d_tr(net, [3, 3, 3, 16], [16, 180, 240, 3], [1, 1, 1, 1],
                    is_training=is_training, scope='Upper_sample/net1')
    net = tf.nn.tanh(net)
    # shape = (none, 180, 240, 3)

    return net


def BiRNN(x):
    batch_size = 256

    cell_1 = tf.nn.rnn_cell.BasicRNNCell(num_units=690, activation=tf.nn.tanh)
    cell_2 = tf.nn.rnn_cell.BasicRNNCell(num_units=690, activation=tf.nn.tanh)

    cells = tf.nn.rnn_cell.MultiRNNCell([cell_1, cell_2])

    initial_state = cells.zero_state(batch_size, dtype=tf.float32)

    outputs, state = tf.nn.dynamic_rnn(cells, x, initial_state=initial_state)
    # print(state)
    return outputs


def loss_GAN(py_x, Y):
    loss = tf.square(py_x - Y)
    # loss = tf.abs(py_x - Y)
    loss = tf.reshape(loss, [16, -1])
    loss = tf.reduce_sum(loss, axis=1)
    loss = tf.reduce_mean(loss)
    return loss


def train():
    with tf.Graph().as_default() as graph:
        x = tf.placeholder(tf.float32, [16, 180, 240, 3], 'x-input')
        y = tf.placeholder(tf.float32, [16, 180, 240, 3], 'y-input')

        X_lower = Lower_sample(x, True)
        X_lower = tf.reshape(X_lower, [256, 16, 690])

        pred = BiRNN(X_lower)
        print(pred)

        upper_data = Upper_sample(pred, True)
        # print(tf.trainable_variables())
        print(upper_data)

        loss_ga = loss_GAN(upper_data, y)
        optimizer_ga = tf.train.AdamOptimizer(0.001)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([tf.group(*update_ops)]):
            train_ga = optimizer_ga.minimize(loss_ga)
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './model_tr/model.cpkt-13100')
        step = 0
        acc_min = 50000.0
        while step * 16 < max_samples:
            if step % display_step == 0:
                batch_x, batch_y = rd.read_data_test()
                loss_g, out = sess.run([loss_ga, upper_data],
                                       feed_dict={x: batch_x, y: batch_y})
                print(step//display_step, loss_g)

                if loss_g < acc_min:
                    print("====================保存模型==================")
                    acc_min = loss_g
                    saver.save(sess, './model_tr/model.cpkt', global_step=step)
                    if os.path.exists('./data/test/model_tr/' + str(step//display_step)):
                        shutil.rmtree('./data/test/model_tr/' + str(step//display_step))
                    os.mkdir('./data/test/model_tr/' + str(step//display_step))
                    out = np.reshape(out, (16, 180, 240, 3))
                    for i in range(len(out)):
                        img_1 = Image.fromarray(np.uint8(batch_y[i] * 127.5 + 127.5))
                        img_2 = Image.fromarray(np.uint8(out[i] * 127.5 + 127.5))
                        img_1.save(
                            './data/test/model_tr/' + str(step//display_step) +
                            '/' + str(2 * i) + '.jpg')
                        img_2.save(
                            './data/test/model_tr/' + str(step//display_step) +
                            '/' + str(2 * i + 1) + '.jpg')
            else:
                batch_x, batch_y = rd.read_data_train()
                sess.run(train_ga,
                         feed_dict={x: batch_x, y: batch_y})
            step += 1


if __name__ == "__main__":
    train()
