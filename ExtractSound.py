#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import subprocess
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

class Video2Sound:
    def __init__(self, path):
        if subprocess.call('which ffmpeg', shell=True):
            '''
            エラーステータスの場合，1が返ってくる
            '''
            raise NotImplementedError('ffmpeg is not installed.')
        self.__video_path = path
        self.__sound = None

    def __call__(self, output=None, to_image=False):
        if not(output):
            path, ext = os.path.splitext(self.__video_path)
            output = path + '.mp3'
        self.__sound = output
        # default bitrate is about 129kbits/s
        cmd = "ffmpeg -y -i {} -loglevel fatal {}".format(self.__video_path, output)
        try:
            res = subprocess.check_output(cmd, shell=True)
            print(res)
        except subprocess.CalledProcessError:
            raise

        if to_image:
            x, fs = self.load_sound()
            mlsp = self.calc_mlsp(x)
            self.save_sound_as_image(mlsp, fs, output)
        return

    def set_path(self, path):
        self.__video_path = path
        self.__sound = None
        return

    def load_sound(self):
        if not(self.__sound):
            raise NotImplementedError('Sound file is not found.')
        '''
        This method will warn that PySoundFile failed loading.
        But this is not fixed because it is not an error.
        To avoid getting this warnings printed on stdout,
        please add below 2 commented lines to this program.
        # import warnings
        # warnings.filterwarnings('ignore') 
        '''
        x, fs = librosa.load(self.__sound, sr=40000)
        return x, fs

    def calc_mlsp(self, x, n_fft=1024, hop_length=128):
        stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
        log_stft = librosa.power_to_db(stft)
        mlsp = librosa.feature.melspectrogram(S=log_stft, n_mels=128)
        return mlsp

    def save_sound_as_image(self, mlsp, fs, output=None, dpi=200):
        path, ext = os.path.splitext(self.__video_path)
        if not(output) or ext!='.jpg':
            output = path + '.jpg'
        librosa.display.specshow(mlsp, sr=fs)
        plt.savefig(output, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        return


if __name__ == '__main__':    
    import argparse
    parser = argparse.ArgumentParser(description='動画ファイルから音声を抽出します。')
    parser.add_argument('-i', '--input',  help='読み込む動画ファイル')
    parser.add_argument('-o', '--output', default='./output.mp3', help='書き出すファイル名')
    arg = parser.parse_args()

    v2s = Video2Sound(arg.input)
    v2s(to_image=True)
