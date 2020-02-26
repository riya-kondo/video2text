#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import pickle, json, os, csv
from collections import defaultdict
import tokenizer

'''
This script creates dataset.pkl from description file.
dictionary based on .pkl file follows as below construction.
["inputs"]: inputs video ids (list of string)
["captions"]: vectorized captions(list of numpy array)
["max_caption_length"]: max caption length(int)
["min_caption_length"]: min caption length(int)
'''

class DatasetCreator:
    def __init__(self, model_dir, caption_file, words_num=8000, corpus_file=None):
        supports_file_types = ['.json', '.csv']
        '''
        The captin file must follow format like bellow:
        If caption_file is json,
          {
            video1: [caption1, caption2, ... captionN],
            video2: [caption1, caption2, ... captionN],
            ...
            videoN: [caption1, caption2, ... captionN],
          }
        If caption_file is csv,
          video1, caption1
          video1, caption2
          ...
          videoN, captionN
        '''
        self.tokenizer = tokenizer.SpTokenizer(model_dir, words_num, corpus_file)
        _, caption_ext = os.path.splitext(caption_file)
        if caption_ext not in supports_file_types:
            raise TypeError('caption_file must be .json or .csv')

        f = open(caption_file, 'r')
        if caption_ext=='.json':
            self.caption_dic = json.load(f)
        elif caption_ext=='.csv':
            reader = csv.reader(f)
            self.caption_dic = defaultdict(list)
            for key, caption in reader:
                self.caption_dic[key].append(caption)
        f.close()
        return 

    def __call__(self, save_path=None):
        pairs = [(vid, caption) for vid, caption in self.caption_dic.items()]
        pairs = [(vid, caption) for vid, caption in zip(self.get_pair(pairs))]

        max_caption_length =  max(len(l) for l in captions)
        min_caption_length =  min(len(l) for l in captions)
        video_ids, captions = zip(*pairs)
        data = {
                'inputs': video_ids,
                'captions': captions,
                'max_caption_length': max_caption_length,
                'min_caption_length': min_caption_length,
               }
        if save_path:
            _, save_ext = os.path.splitext(save_path):
            if save_ext!='.pkl':
                save_path += '.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
        return data

    def get_pair(self, pairs:list):
        ids = []
        sentences = []
        for vid, captions in pairs:
            captions = [self.tokenizer.get_bos() + self.tokenizer.words2ids(c) + self.tokenizer.get_eos()
                        for c in captions]
            sentences.extend(captions)
            videos += [vpath]*len(captions)
        assert len(ids)==len(sentences), 'ids and sentences length is difference'
        return ids, sentences


if __name__ == '__main__':
    import time, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus_file', help='path to corpus raw text file')
    parser.add_argument('--model_dir', help='path to model file(*.model)')
    parser.add_argument('--caption_file', help='path to caption file(*.json or *.csv)')
    parser.add_argument('-w', '--words', default=8000, help='max word snumber of tokenize')

    args = parser.parse_args()
    
    model_dir = os.path.abspath(args.model_dir) 
    
    creator = DatasetCreator(model_dir, caption_file, args.words, args.corpus_file)

    save_path = os.path.join(model_dir, 'dataset.pkl')
   
    st = time.time() 
    creator(save_path)
    print('Time Elapsed {} sec to create dataset'.format(time.time() - st))
