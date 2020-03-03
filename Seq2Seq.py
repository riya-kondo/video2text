#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import tensorflow as tf
from BasicEncoder from Encoders
from BasicDecoder from Decoders


class Seq2Seq(tf.keras.models.Model):
    def __init__(self, encoder:BasicEncoder, decoder:BasicDecoder,
                 max_caption_length):
        super()__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.__max_caption_length = max_caption_length

    def __call__(self, x, y=None, train=True):
        if train:
            output = self.train(x, y)
            return output
        else:
            output, attention_weight = self.generate(x, y)
            return output, attention_weight

    @tf.function
    def train(self, x, y):
        hiddens, encoder_output = self.encoder(x)
        dec_inputs = y[:, 0]
        outputs = []
        for i in range(1, y.shape[1]):
            dec_outputs, hiddens, _ = self.decoder(dec_inputs,
                                                   hiddens,
                                                   train=True,
                                                   encoder_output=encoder_output)
            outpus.append(dec_outputs)
            dec_inputs = y[:,i]
        return outputs

    @tf.function
    def generate(self, x, dec_inputs):
        hiddens, encoder_output = self.encoder(x)
        outputs = []
        for i in range(self.__max_caption_length):
            dec_outputs, hiddens, attention_weight = self.decoder(dec_inputs,
                                                                  hiddens,
                                                                  train=False,
                                                                  encoder_output=encoder_output)
            outputs.append(dec_outputs)
            dec_inputs = dec_outputs
        return outputs, attention_weight


if __name__ == '__main__':
    pass
