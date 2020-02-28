#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import subprocess
import argparse

class Video2Sound:
    def __init__(self, path):
        if subprocess.call('which ffmpeg', shell=True):
            '''
            エラーステータスの場合，1が返ってくる
            '''
            raise NotImplementedError('ffmpeg is not installed.')
        self.__video_path = path

    def __call__(self, output=None):
        if not(output):
            path, ext = os.path.splitext(self.__video_path)
            output = path + '.mp3'
        # default bitrate is about 129kbits/s
        cmd = "ffmpeg -y -i {} -loglevel fatal {}".format(self.__video_path, output)
        try:
            res = subprocess.check_output(cmd, shell=True)
            print(res)
        except subprocess.CalledProcessError:
            raise
        return

    def set_path(self, path):
        self.__video_path = path
        return



if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='動画ファイルから音声を抽出します。')
    parser.add_argument('-i', '--input',  help='読み込む動画ファイル')
    parser.add_argument('-o', '--output', default='./output.mp3', help='書き出すファイル名')
    arg = parser.parse_args()

    v2s = Video2Sound(arg.input)
    v2s()
