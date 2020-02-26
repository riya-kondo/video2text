#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf


class BasicEncoder(tf.keras.models.Model):
    def __init__(self, enc_units, batch_sz, RNNLayer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.mask = tf.keras.layers.Masking(mask_value=0., input_shape=(,self.enc_units))
        self.rnn = RNNLayer

    def call(self, x, hidden):
        x = self.mask(x)
        output, state = self.rnn(x, initial_state=hidden)
        return output, state

    def initialize_state(self, batch_size=None):
        if not(batch_size):
            batch_size = self.batch_sz
        # init_state is 2 length list because lstm needs 2 states(cell and hidden). 
        init_state = [tf.zeros((batch_size, self.enc_units)) for i in range(2)]
        return init_state


class StackedEncoder(BasicEncoder):
    def __init__(self, enc_units, batch_sz, RNNLayer, stack_depth, *args, **kwargs):
        super().__init__(enc_units, batch_sz, RNNLayer, *args, **kwargs)
        self.stacked_rnn = [self.rnn for i in range(stack_depth)]

    def call(self, x: tf.Tensor, hidden:list):
        x = self.mask(x)
        states = []
        for layer in self.stacked_rnn:
            x, state = layer(x, initial_state=hidden)
            states.append(state)
        return x, states


class BidirectionalEncoder(BasicEncoder):
    def __init__(self, enc_units, batch_sz, stack_num=1, *args, **kwargs):
        BiLSTMLayer = tf.keras.layers.LSTM(enc_units, return_sequences=True,
                                           return_state=True, go_backward=True,
                                           recurrent_initializer='glorot_uniform')
        LSTMLayer = tf.keras.layers.LSTM(enc_units, return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')
        super().__init__(enc_units, batch_sz, *args, **kwargs)
        self.stack_num = stack_num
        self.f_rnns = [LSTMLayer for i in range(self.stack_num)]
        self.b_rnns = [BiLSTMLayer for i in range(self.stack_num)]

    def call(self, x:tf.Tensor, hidden:list):
        x = self.mask(x)
        f_ = b_ = x
        for i in range(self.stack_num):
            f_, f_state_h, f_state_c = self.f_rnns[i](f_)
            b_, b_state_h, b_state_c = self.b_rnns[i](b_)
        output = tf.concat([f_, b_], axis=-1)
        states = [f_state_h, f_state_c]
        return output, states 
