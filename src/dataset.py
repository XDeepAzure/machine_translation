#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：machine_translation 
@File    ：dataset.py
@Author  ：湛蓝
@Date    ：2022/8/18 17:28 
'''

#@save
import os

import torch
from d2l import torch as d2l
from torch.utils.data import Dataset

from vocab import Vocab

d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

# dt, _, _ = d2l.load_data_nmt(3, 10)
# i = 0
# for d in dt:
#     i+=1
# print(i)

#@save
def read_data_nmt():
    """载入“英语－法语”数据集"""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
             encoding='utf-8') as f:
        return f.read()

# raw_text = read_data_nmt()
# print(raw_text[:75])

def preprocess_nmt(text:str):
    def no_space(char, prev_char):              #判断单词与标点之间有无空格
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格 \u202f与\xa0
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i-1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)
# text = preprocess_nmt(raw_text)
# print(text[:80])
def tokenize_nmt(text:str, num_examples=None):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return (source, target)
# raw_text = read_data_nmt()
# text = preprocess_nmt(raw_text)
# source, target = tokenize_nmt(text)
# # for i in range(6):
# #     print(source[i], '\t', target[i])
# src_vocab = Vocab(source, min_freq=2,
#                       reserved_tokens=['<pad>','<bos>','<eos>'])
# # src_vocab = d2l.Vocab(source, min_freq=2,
# #                       reserved_tokens=['<pad>','<bos>','<eos>'])
# print(len(src_vocab))

def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))

# s = truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
# print(s)

def build_array_nmt(lines, vocab, num_steps):
    '''
    这部分应用在Dataset中是在init中增加这样一个操作，先给每条句子加上<eos>结尾然后填充截断
    :param lines:
    :param vocab:
    :param num_steps:
    :return:
    '''
    #将word转换成indices
    lines = [vocab[l] for l in lines]
    #为每一条语句加上eos结尾
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines
    ])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len

def load_data_nmt(num_examples=600):
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text,num_examples)
    src_vocab = Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    #转换成indices
    source = [[src_vocab[token] for token in line] for line in source]
    target = [[tgt_vocab[token] for token in line] for line in target]
    return source, target, src_vocab, tgt_vocab

class translationDataset(Dataset):
    def __init__(self, num_steps, batch_size, device, num_examples=600):
        super().__init__()
        self.num_steps, self.batch_size = num_steps, batch_size
        self.source, self.target, self.src_vocab, self.tgt_vocab = load_data_nmt(num_examples=num_examples)
        #source中每个句子长度不一，所以暂时不能做转成tensor
        # self.source = torch.tensor(self.source, dtype=torch.int64)
        self.source = [l + [self.src_vocab['<eos>']] for l in self.source]
        for i in range(len(self.source)):
            self.source[i] = truncate_pad(self.source[i], self.num_steps, self.src_vocab['<pad>'])
        self.source = torch.tensor(self.source, dtype=torch.int32, device=device)

        self.target = [l + [self.tgt_vocab['<eos>']] for l in self.target]
        for i in range(len(self.target)):
            self.target[i] = truncate_pad(self.target[i], self.num_steps, self.tgt_vocab['<pad>'])
        self.target = torch.tensor(self.target, dtype=torch.int32, device=device)


    def __getitem__(self, index):
        #考虑句子的截断和填充实在init中实现还是在此处实现
        source, target = self.source[index], self.target[index]
        src_valid_len = (source != self.src_vocab['<pad>']).type(torch.int32).sum()
        tat_valid_len = (target != self.tgt_vocab['<pad>']).type(torch.int32).sum()
        return source, src_valid_len, target, tat_valid_len
        pass

    def __len__(self):
        return min(len(self.source), len(self.target)) // self.batch_size * self.batch_size


if __name__ == '__main__':
    dataset = translationDataset(3, 10, d2l.try_gpu())
    print(len(dataset))
    # for i in range(5):
    #     s, sl, t, tl = dataset.__getitem__(i)
    #     print(s[:sl], dataset.src_vocab.to_tokens(s[:sl].tolist()))
    #     print(t[:tl], dataset.tgt_vocab.to_tokens(t[:tl].tolist()))
    pass
