#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import cv2
import os

class VideoSplitter:
    '''
    self.path: videoまでのパス
    self.video_name: videoのファイル名（拡張子含む）
    self.__capture: VideoCaptureオブジェクト
    '''
    def __init__(self, path):
        '''
        path: videoまでのパス
        '''
        self.path = os.path.abspath(path)
        _, self.ext = os.path.splitext(path)
        self.video_name = os.path.basename(path).replace(self.ext, '')
        self.__capture = cv2.VideoCapture(self.path)
        if not(self.__capture.isOpened()):
            raise FileNotFoundError('Specified path is not video')
        self.fps = self.__capture.get(cv2.CAP_PROP_FPS)
        self.height = self.__capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width = self.__capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frames = None

    def __del__(self):
        self.__capture.release()

    def get_frames(self):
        if not(self.frames):
            self.frames = self._split_video()
        return self.frames

    def _split_video(self):
        '''
        生成したVideoCaptureオブジェクトをフレーム毎に分割してreturnします。
        '''
        frames = []
        while True:
            ret, frame = self.__capture.read()
            if ret==True:
                print(True)
                frames.append(frame)
            else:
                print(False)
                break
        return frames

    def save_frames(self, path):
        '''
        フレームのリストを指定したパスにjpgファイルで保存します。
        path: フレームファイル(jpg)を保存するディレクトリ
        '''
        if not(self.frames):
            self.frames = self._split_video()

        # 指定したディレクトリが存在しなければ作成する。
        if not(os.path.isdir(path)):
            os.makedirs(path)

        for i, frame in enumerate(self.frames):
            file_name = os.path.join(path, self.video_name + '_%05d.jpg' %(i)
            cv2.imwrite(file_name, frame)
        return


if __name__ == '__main__':
    # Usage:
    import sys
    video_path = os.path.abspath(sys.argv[1])
    splitter = VideoSplitter(video_path) 
    frames = splitter.get_frames()
    import pdb; pdb.set_trace()
