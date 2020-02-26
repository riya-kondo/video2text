#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from abc import ABCMeta, abstractmethod
import pickle as pkl

class Tokenizer(metaclass=ABCMeta):
    def __init__(self, model):
        self.model = model
    
    @abstractmethod
    def ids2words(self, ids):
        pass

    @abstractmethod
    def words2ids(self, words):
        pass
    
    @abstractmethod
    def get_bos(self):
        pass

    @abstractmethod
    def get_eos(self):
        pass

    @abstractmethod
    def get_vocab_size(self):
        pass


class TfTokenizer(Tokenizer):
    def __init__(self, path_to_pkl:str=None, words_num=8000, path_to_corpus:str=None):
        model = self._get_model(path_to_pkl, words_num, path_to_corpus)
        super().__init__(model)

    def ids2words(self, ids):
        if type(ids[0]) is int:
            words = self.model.sequences_to_texts([ids])
        else:
            words = self.model.sequences_to_texts(ids)
        return words

    def words2ids(self, words):
        if type(words) is str:
            ids = self.model.texts_to_sequences([words])
        else:
            ids = self.model.texts_to_sequences(words)
        return ids

    def get_bos(self):
        return self.model.word_index['<start>']

    def get_eos(self):
        return self.model.word_index['<end>']
    
    def get_unk(self):
        return self.model.word_index['<unk>']

    def get_vocab_size(self):
        return self.model.num_words

    def _get_model(self, path_to_pkl, words_num, path_to_corpus):
        if path_to_pkl:
            with open(path_to_pkl, 'rb') as f:
                model = pkl.load(f)
        else:
            with open(path_to_corpus, 'r') as f:
                corpus = f.readlines()
            model = tf.keras.preprocessing.text.Tokenizer(num_words=num_words,
                                                          oov_token='<unk>')
            model.fit_on_texts(corpus)
        return model


class SpTokenizer(Tokenizer):
    def __init__(self, model_dir:str, words_num=8000, path_to_corpus:str=None):
        import sentencepiece as spm
        sp = self._get_sentencepiece(model_dir, words_num, path_to_corpus)
        super().__init__(sp)

    def ids2words(self, ids):
        if type(ids[0]) is int:
            words = [self.model.decode_ids(ids)]
        else:
            words = [self.model.decode_ids(i) for i in ids]
        return words

    def words2ids(self, words):
        if type(words) is str:
            ids = [self.model.encode_as_ids(words)]
        else:
            ids = [self.model.encode_as_ids(w) for w in words]
        return ids

    def get_bos(self):
        return self.model.bos_id()

    def get_eos(self):
        return self.model.eos_id()

    def get_vocab_size(self):
        return len(self.model)

    def _get_sentencepiece(model_dir, word_num=8000, path_to_corpus:str=None):
        sp = spm.SentencePieceProcessor()
        if path_to_corpus:
            '''
            Load Sentencepiece with Training.
            '''
            if not(os.path.exists(model_dir)):
                os.mkdir(model_dir)
            prefix = os.path.join(model_dir, 'model')
            train_arg = '--input={} --model_prefix={} --vocab_size={} --character_coverage=1.0'
            train_arg += ' --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3'
            train_arg += ' --pad_piece=<pad> --unk_piece=<unk> --bos_piece=<start> --eos_piece=<end>'
            spm.SentencePieceTrainer.Train(train_arg.format(path_to_corpus, prefix, word_num))
            sp.Load(prefix+'.model')
        else:
            '''
            Load Sentencepiece from pretrained model.
            '''
            try:
                sp.Load(glob.glob(model_dir+'/*.model')[0])
            except IndexError:
                print('specified model directory must have .model file.')
        return sp

if __name__ == '__main__':
    import sys
    import os

    if sys.argv[1][-3:]=='pkl':
        f = open(sys.argv[1], 'rb')
        data = pkl.load(f)
        f.close() 
        tokenizer = TfTokenizer(data['tokenizer'])
    elif sys.argv[1][-5:]=='model':
        tokenizer = SpTokenizer(sys.argv[1])
    else:
        raise TypeError(sys.argv[1][:-5])
    ids = [   2,    4,    8,    6, 3367,   18,    7,   75,   11,  203,  349,
          5,    3,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0]
    words = tokenizer.ids2words(ids)
    print(words)
