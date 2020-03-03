#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import tensorflow as tf


class BasicDecoder(tf.keras.models.Model):
    def __init__(self, embedding_dim, units, vocab_size, batch_sz,
                 dropout, RNNLayer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_sz = batch_sz
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.rnn = RNNLayer
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)

    def call(self, x, hiddens, train=True, **kwargs):
        x = self.embedding(x)
        initial_states = hiddens
        output, state_h, state_c = self.lstm(x, initial_state=initial_states)
        output = self.dropout(output, training=train)
        x = self.fc(output)
        states = [state_h, state_c]
        return x, states

    def reset_state(self):
        init_state = [tf.zeros((self.batch_sz, self.units)) for i in range(2)]
        return init_state


class AttentionDecoder(BasicDecoder):
    def __init__(self, embedding_dim, units, vocab_size, batch_sz, 
                 dropout, RNNLayer, AttentionLayer, *args, **kwargs):
        super().__init__(embedding_dim, units, vocab_size, 
                         batch_sz, dropout, RNNLayer, *args, **kwargs)
        self.attention = AttentionLayer

    def call(self, x, hiddens, train=True, **kwargs):
        context_vector, attention_weight = self.attention(hiddens[0],
                                                          kwargs['enc_output'])
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector,1), x], axis=-1)
        initial_states = hiddens
        output, state_h, state_c = self.lstm(x, initial_state=initial_states)
        output = self.dropout(output, training=train)
        x = self.fc(output)
        states = [state_h, state_c]
        return x, states, attention_weight

    def reset_state(self):
        init_state = [tf.zeros((self.batch_sz, self.units)) for i in range(2)]
        return init_state
