#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import tensorflow as tf


class FeatureExtractor:
    def __init__(self, batch_sz=1, max_frame=80):
        self.model = self._create_model()
        self.batch_sz = batch_sz
        # set to avoid OOM
        self.max_frame = max_frame

    def _create_model(self):
        '''
        build the model for extract feature
        '''
        net = tf.keras.applications.InceptionV3(include_top=True,
                                                weights='imagenet')
        for layers in net.layers:
            layers.trainable = False
        input_layer = net.input
        hidden = net.layers[-2].output
        extract_model = tf.keras.Model(input_layer, hidden)
        return extract_model

    def __call__(self, inputs):
        '''
        inputs:[frame length, height, width, channel] tensor or path to img
        Or inputs is list of paths to img.
        '''
        if type(inputs) is str:
            inputs = self.load_images(inputs)

        if type(inputs[0]) is str:
            inputs = [self.load_images(path) for path in inputs]

        if type(inputs) is list:
            inputs = tf.concat(inputs, axis=0)
            
        if inputs.shape[0] > self.max_frame:
            feats = self._get_feats_loop(inputs)
        else:
            feats = self.model(inputs)
        return feats # [frame length, feat_dim(2048)]

    def _get_feats_loop(self, inputs):
        '''
        フレーム数がmax_frame以上の場合は，OOMを避けるためにループ処理で変換します．
        '''
        feats_of_frames = []
        for i in inputs:
            feature = self.model(tf.expand_dims(i, 0))
            feats_of_frames.append(feature)
        feats_of_frames = tf.concat(feats_of_frames, axis=0)
        return feats_of_frames

    def load_images(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299,299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        img = tf.expand_dims(img, 0)
        return img
    
    def load_from_array(self, array):
        '''
        Load image from array which shape is (width, height, rgb)
        '''
        img = tf.image.resize(array, (299,299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        img = tf.expand_dims(img, 0)
        return img


if __name__ == '__main__':
    import sys, glob, os, time
    from tqdm import tqdm
    '''
    usage: python FeatureExtractor.py path/to/directory
    You can specify directory which includes images or which includes directory.
    '''
    fe = FeatureExtractor()
    dirs = sorted(glob.glob(sys.argv[1]+'/*'))
    start = time.time()
    if dirs[0][-4:]=='.jpg':
        files = dirs
        images = [fe.load_images(f) for f in files]
        feats = fe(images)
    else:
        for d in tqdm(dirs):
            files = sorted(glob.glob(d+'/*.jpg'))
            images = [fe.load_images(f) for f in files]
            feats = fe(images)
    print('time elapsed {} sec'.format(time.time() - start))

