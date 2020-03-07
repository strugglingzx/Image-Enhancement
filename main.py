#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/5/6 19:11
# @Author  : Xin Zhang
# @File    : main.py

from network import Filtering, log_loss
from data_load import image_load
import tensorflow as tf
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import scipy.signal
from PIL import Image
from config import *
import os


def change(batch_of_imgs, h, w):
    batch_input = []
    for t in range(batch_of_imgs.shape[0]):
        t_b = np.max(batch_of_imgs[t], axis=2)
        t_b = np.reshape(t_b, (h, w, 1))
        batch_img = np.append(batch_of_imgs[t], t_b, axis=2)
        batch_input.append(batch_img)
    batch_input = np.array(batch_input)
    batch_input = batch_input.astype(np.float32)
    return batch_input


def train(pathone, pathtwo, path1, path2):
    inputs_ = tf.placeholder(tf.float32, shape=[None, None, None, 4])
    targets_ = tf.placeholder(tf.float32, shape=[None, None, None, 3])

    net = Filtering(inputs_, 3)
    output = net.filter_generater()

    loss = log_loss(output, targets_)
    tf.summary.scalar('loss', loss)
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(lr, global_step, 100, decay_rate=0.98, staircase=True)
    # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver()
    dataset = image_load(pathone, pathtwo, image_height, image_width, batch_size, 3, shuffle=True)
    data = dataset.batch_generator()
    train_data = data.make_one_shot_iterator()
    next_element = train_data.get_next()

    evalset = image_load(path1, path2, image_height, image_width, batch_size, 3, shuffle=True)
    eval_ = evalset.batch_generator()
    eval_data = eval_.make_one_shot_iterator()
    eval_element = eval_data.get_next()

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    merged_summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        # saver.restore(sess, test_path)
        totalloss = []
        best_loss = 999.0
        sess.run(init_op)
        writer = tf.summary.FileWriter("log/", sess.graph)
        for i in range(step):
            batch_of_imgs, label = sess.run(next_element)

            batch_input = change(batch_of_imgs, image_height, image_width)

            cost, _, ta, merge, pred = sess.run([loss, train_op, targets_, merged_summary_op, output],
                                                feed_dict={inputs_: batch_input, targets_: label})
            writer.add_summary(merge, i)
            totalloss.append(cost)
            print("Step {:3d} | Epoch {:3d} | ".format(i + 1, i//880 + 1), "Training loss  {:.6f}".format(cost))
            if i % 100 == 0:
                eval_imgs, eval_label = sess.run(eval_element)
                eval_input = change(eval_imgs, image_height, image_width)
                result, eval_loss = sess.run([output, loss], feed_dict={inputs_: eval_input, targets_: eval_label})
                print('Evaluation loss   {:.6f}'.format(eval_loss))

                image1 = np.resize(result[0], [image_height, image_width, 3])
                scipy.misc.toimage(image1, cmin=0.0, cmax=1.0).save(save_dir + str(i) + '.jpg')

                saver.save(sess, save_path)

        saver.save(sess, save_path)
        x = range(0, len(totalloss))
        plt.plot(x, totalloss)
        plt.savefig(save_dir + 'loss.jpg')


def test(path_t):

    inputs_ = tf.placeholder(tf.float32, shape=[None, None, None, 4])

    net = Filtering(inputs_, 3)
    output = net.filter_generater()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, test_path)
        for j, path in enumerate(path_t):
            (fp, t) = os.path.split(path)
            (fn, ext) = os.path.splitext(t)
            image = Image.open(path)
            img_array = np.array(image)
            [m, n, t] = img_array.shape
            img = img_array.astype(np.float32) / 255.0
            img = np.reshape(img, (1, m, n, 3))
            batch_input = change(img, m, n)

            result = sess.run([output], feed_dict={inputs_: batch_input})
            print(result[0][0].shape)
            for i in range(result[0].shape[0]):
                image1 = np.resize(result[0][i], [m, n, 3])
                out = image1 * 1.1
                scipy.misc.toimage(out, cmin=0.0, cmax=1.0).save(test_save + fn + '.jpg')


if __name__ == '__main__':
    dir1 = glob.glob(dir)
    dir2 = glob.glob(dir_label)
    dir3 = glob.glob(dir_eval)
    dir4 = glob.glob(dir_eval_label)

    dir_t = glob.glob(dir_test)
    if mode == 'train':
        train(dir1, dir2, dir3, dir4)
    elif mode == 'test':
        test(dir_t)
