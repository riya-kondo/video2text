#! /usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import glob
import numpy as np
import time
import scipy.fftpack
import cv2
import pdb
from PIL import Image

        
class hashConverter:
    def __init__(self, images=None, video_name=None, hash_size=8, threshold=12):
        if type(images[0]) is str:
            self.frame_list = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in images]
        else:
            self.frame_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
        self.video_name = video_name
        self.hash_size = hash_size
        self.threshold = threshold

    def __call__(self):
        '''
        以下のように，シーンごとのシーケンス番号を返します．
        [[0,1,...,47,48],[49,50,...,87,88],[89,90,...135,136]]
        フラッシュや暗転のシーンのシーケンスは，detection()関数で削除されています．
        '''
        prev_frame = 0
        start_frame = 0
        segments = []
        scene = [0]

        for i in range(1, len(self.frame_list)):
            next_frame = i
            diff = self.calc_hamming_distance(i-1, i)
            if diff > self.threshold:
                if self.detection(prev_frame) and self.detection(start_frame):
                    start_frame = next_frame
                    scene = []
                else:  
                    segments.append(scene)
                    scene = [i]
                    break
            prev_frame = next_frame
            scene.append(i)

        i += 1
        if i == len(self.frame_list):
            if scene:
                segments.append(scene)
                return segments
            else:
                return None

        prev_frame = next_frame
        start_frame = next_frame 

        for j in range(i, len(self.frame_list)):
            next_frame = j
            diff = self.calc_hamming_distance(prev_frame, next_frame)
            if diff > self.threshold:
                if self.detection(prev_frame) and self.detection(start_frame):
                    prev_frame = segments[-1][-1]
                    diff = self.calc_hamming_distance(prev_frame, next_frame)
                    if diff > self.threshold:
                        prev_frame = next_frame
                        start_frame = next_frame
                        scene = []
                    else:
                        scene = segments.pop()
                        start_frame = scene[0]
                        prev_frame = next_frame
                else:  
                    segments.append(scene)
                    prev_frame = next_frame
                    start_frame = next_frame
                    scene = []
            else:
                prev_frame = next_frame
            scene.append(j)
        segments.append(scene)
        return segments 

    def generate_hash(self, index):
        '''
        画像のシーケンス番号を渡すとハッシュ値を返す
        返り値: (hash(binary))
        '''
        raise NotImplementedError
        pass

    def get_video_name(self):
        return self.video_name

    def detection(self, index):
        '''
        フラッシュ・フェード検出
        引数 index: 画像のシーケンス番号
        返り値 (bool): フラッシュまたはフェードフレームかどうか
        '''
        img = self.frame_list[index]
        mean = np.mean(img/255)
        std = np.std(img/255)
        if mean < 0.2 or 0.8 < mean:
            if std <= 0.05:
                return True 
        return False


    def calc_hamming_distance(self, index1, index2):
        '''
        2つのindex(シーケンス番号)からハミング距離を計算する。
        hash1, hash2は同bitの２進数(str)
        '''
        hash1 = self.generate_hash(index1)
        hash2 = self.generate_hash(index2)
        diff = np.sum(np.not_equal(hash1, hash2))
        return diff


class AhashConverter(hashConverter):
    def __init__(self, images, video_name, hash_size=8, threshold=10):
        super().__init__(images, video_name, hash_size, threshold)

    def generate_hash(self, index):
        '''
        # tensorflow ver.
        img = tf.image.resize(tf.expand_dims(img, -1), (self.hash_size, self.hash_size))
        imgarray = tf.reshape(img, (self.hash_size**2,))
        mean = tf.reduce_mean(imgarray)
        mask = tf.math.less(imgarray, mean)
        hash_num = tf.cast(mask, dtype=tf.int32) * tf.ones(self.hash_size**2, dtype=tf.int32) 
        '''
        img = self.frame_list[index]
        img = cv2.resize(img, (self.hash_size, self.hash_size))
        imgarray = np.array(img, dtype=np.float).reshape(self.hash_size**2)
        mean = imgarray.mean()
        hash_num = np.where(imgarray < mean, 1, 0)

        return hash_num


class PhashConverter(hashConverter):
    def __init__(self, images, video_name, hash_size=32, threshold=10):
        if hash_size < 8:
            raise ValueError('Hash size must be larger than 8')
        super().__init__(images, video_name, hash_size, threshold)

    def generate_hash(self, index):
        '''
        perceptive Hash alghorythm
        離散コサイン変換を行い，画像の低周波成分を用いてハッシュ化を行います．
        refer this: http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
        '''
        pdb.set_trace()
        img = self.frame_list[index]
        img = cv2.resize(img, (self.hash_size, self.hash_size))
        imgarray = np.array(img, dtype=np.float).reshape(self.hash_size**2)
        dct = scipy.fftpack.dct(imgarray)
        dct = np.reshape(dct, (self.hash_size, self.hash_size))[:8, :8]
        mean = dct.mean()
        hash_num = np.where(dct < mean, 1, 0)
        return hash_num


class DhashConverter(hashConverter):
    def __init__(self, images, video_name, hash_size=8, threshold=10):
        super().__init__(images, video_name, hash_size, threshold)

    def generate_hash(self, index=0):
        '''
        differencive hash alghorythm 
        隣接するピクセルの差分をとって，ハッシュ化を行います．
        '''
        img = self.frame_list[index]
        img = cv2.resize(img, (self.hash_size+1, self.hash_size))
        imgarray = np.array(img, dtype=np.float).reshape(self.hash_size+1, self.hash_size)
        hash_num = imgarray[1:,:] > imgarray[:-1,:]
        pdb.set_trace()
        return hash_num


class NNConverter(hashConverter):
    def __init__(self, images, video_name, hash_size=8, threshold=10):
        super().__init__(images, video_name, hash_size, threshold)
        import tensorflow as tf
        self.model = self._setup_model()


    def generate_nn(self, index=None, path=None):
        '''
        .npyへのパスを渡すと特徴量を返す
        返り値: numpy shape(2048)
        '''
        if path:
            return np.load(path)[index]
        img = self.preprocess_input(self.frame_list[index])
        feats = self.model(img)
        return feats

    def preprocess_input(self, array):
        img = tf.image.resize(array, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        img = tf.expand_dims(img, 0)
        return img

    def _setup_model(self):
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

    def calc_cos_similarity(self, v1, v2):
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        sim = dot / norm
        return sim

if __name__ == '__main__':
    import sys, glob, pdb
    path_to_frames_dir = os.path.abspath(sys.argv[1])
    vid = os.path.basename(path_to_frames_dir)
    files = sorted(glob.glob(path_to_frames_dir+'/*.jpg'))
    frames = [cv2.imread(f) for f in files]
    cv = DhashConverter(frames, vid)
    segments = cv()
    hashs = [cv.generate_hash(f) for f in range(len(frames))]
    pdb.set_trace()
    print(segments)
