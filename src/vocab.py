#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：machine_translation 
@File    ：vocab.py
@Author  ：湛蓝
@Date    ：2022/8/20 21:43 
'''
import collections


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        '''
        构建字典，word2index、index2word
        :param tokens:
        :param min_freq:
        :param reserved_tokens:保留token如<eos>
        '''
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x:x[1], reverse=True)
        self.idx2token = ['<unk>'] + reserved_tokens\
                         + [token for token, freq in self._token_freqs if freq>=min_freq]
        self.token2idx = {
            token: idx for idx, token in enumerate(self.idx2token)
        }
    def __len__(self):
        return len(self.idx2token)

    def __getitem__(self, tokens):
        '''暂时不清楚这个函数的用处'''
        if not isinstance(tokens, (list, tuple)):
            #若token存在返回idx，反之返回unk
            return self.token2idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx2token[indices]
        return [self.idx2token[index] for index in indices]

    @property
    def unk(self):#未知次元的索引
        return self.token2idx['<unk>']

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):
    '''
    统计词元频率
    :param tokens: list[list[]]
    :return:
    '''
    if len(tokens) == 0 or isinstance(tokens[0], list):
        #将词元列表展开成一个一维的
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

if __name__ == '__main__':
    pass
