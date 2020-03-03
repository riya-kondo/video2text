#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import tensorflow as tf


class MultiheadAttention(tf.keras.models.Model):
    def __init__(self, depth: int, head_num: int, dropout_rate: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_dim = depth
        self.head_num = head_num
        self.dropout_rate = dropout_rate

        self.q_dense_layer = tf.keras.layers.Dense(depth, use_bias=False)
        self.k_dense_layer = tf.keras.layers.Dense(depth, use_bias=False)
        self.v_dense_layer = tf.keras.layers.Dense(depth, use_bias=False)
        self.output_dense_layer = tf.keras.layers.Dense(depth, use_bias=False)
        self.attention_dropout_layer = tf.keras.layers.Dropout(dropout_rate)

    def call(self, query: tf.Tensor, memory: tf.Tensor, attention_mask: tf.Tensor, training: bool) -> tf.Tensor:
        q = self.q_dense_layer(query)
        k = self.k_dense_layer(memory)
        v = self.v_dense_layer(memory)
        
        q = self._split_head(q)
        k = self._split_head(k)
        v = self._split_head(v)
        
        depth = self.hidden_dim // self.head_num
        q *= depth ** -0.5
        
        logit = tf.matmul(q, k, transpose_b=True)
        logit += tf.cast(attention_mask, dtype=tf.float32) * query.dtype.min

        attention_weight = tf.nn.softmax(logit)
        attention_weight = self.attention_dropout_layer(attention_weight, training=training)

        attention_output = tf.matmul(attention_weight, v)
        attention_output = self._combine_head(attention_output)
        context_vector = self.output_dense_layer(attention_output)
        return context_vector, attention_weight

    def _split_head(self, x: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('split_head'):
            batch_size, length, hidden_dim = tf.unstack(tf.shape(x))
            x = tf.reshape(x, [batch_size, length, self.head_num, self.hidden_dim // self.head_num])
            return tf.transpose(x, [0, 2, 1, 3])
            
    def _combine_head(self, x: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('combine_head'):
            batch_size, _, length, _ = tf.unstack(tf.shape(x))
            x = tf.transpose(x, [0, 2, 1, 3])
            return tf.reshape(x, [batch_size, length, self.hidden_dim])


class SelfAttention(MultiheadAttention):
    def call(self, input: tf.Tensor, attention_mask: tf.Tensor, training: bool) -> tf.Tensor:
        return super().call(input=input, memory=input, attention_mask=attention_mask, training=training)


class SimpleAttention(tf.keras.models.Model):
    def __init__(self, depth: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth = depth

        self.q_dense_layer = tf.keras.layers.Dense(depth, use_bias=False)
        self.k_dense_layer = tf.keras.layers.Dense(depth, use_bias=False)
        self.v_dense_layer = tf.keras.layers.Dense(depth, use_bias=False)
        self.output_dense_layer = tf.keras.layers.Dense(depth, use_bias=False)

    def call(self, input_tensor: tf.Tensor, memory: tf.Tensor) -> tf.Tensor:
        q = self.q_dense_layer(input_tensor)
        k = self.k_dense_layer(memory)
        v = self.v_dense_layer(memory)

        q *= self.depth ** -0.5 # with scaling
        logit = tf.matmul(q, k, transpose_b=True)

        attention_weight = tf.nn.softmax(logit)

        attention_output = tf.matmul(attention_weight, v)
        return self.output_dense_layer(attention_output)


class BahdanauAttention(tf.keras.models.Model):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis))

        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class LuongAttention(tf.keras.models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, query, values):
        '''
        query: decoder hidden(batch_size, units)
        values: encoder output(batch_size, max_length, units)
        '''
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = tf.matmul(hidden_with_time_axis, values, transpose_b=True) # calc dot product
        score = tf.squeeze(score, [1]) # reduce dimension
        attention_weights = tf.nn.softmax(score, axis=1) # apply softmax to dot product
        context_vector = tf.expand_dims(attention_weights,2) * values
        context_vector = tf.reduce_sum(context_vector, axis=1) # calc weighted average
        return context_vector, attention_weights

if __name__ == '__main__':
    batch_sz = 2
    units_num = 4
    length = 3
    att = BahdanauAttention(4)
    sample_query = tf.random.normal((batch_sz, 3))
    sample_value = tf.random.normal((batch_sz, length, units_num))
    context = att(sample_query, sample_value)
    import pdb;pdb.set_trace()
