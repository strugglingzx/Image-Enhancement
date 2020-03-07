#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/5/6 18:38
# @Author  : Xin Zhang
# @File    : network.py
import tflearn
import tensorflow as tf
from config import *


class Filtering(object):
    def __init__(self, input_, channel_out):
        self.input_ = input_
        self.channel = channel_out

    def filter_generater(self):
        n = 3
        net = tflearn.conv_2d(self.input_[:, :, :, :3], 16, 3, activation='prelu')
        net = tflearn.residual_block(net, n, 16, activation='prelu')
        net = tflearn.residual_block(net, n, 32, activation='prelu')
        net = tflearn.residual_block(net, n, 32, activation='prelu')
        net = tflearn.conv_2d(net, self.channel, 3, activation='prelu')
        net = net * tf.concat(tf.expand_dims(self.input_[:, :, :, 3], 3) * 3, axis=3)
        return net

    def filter_generater_nomap(self):
        n = 3
        net = tflearn.conv_2d(self.input_[:, :, :, :3], 16, 3, activation='prelu')
        net = tflearn.residual_block(net, n, 32, activation='prelu')
        net = tflearn.residual_block(net, n, 32, activation='prelu')
        net = tflearn.residual_block(net, n, 32, activation='prelu')
        net = tflearn.conv_2d(net, self.channel, 3, activation='prelu')
        net = net * self.input_[:, :, :, :3]
        return net


def log_loss(output, truth):
    loss = 0.5 * (tf.reduce_mean(tf.pow((truth - output), 2))
                  + 1 - tf.reduce_mean(tf.image.ssim(truth, output,  max_val=1.0)))
    return loss


