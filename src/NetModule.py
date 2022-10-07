#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：machine_translation 
@File    ：NetModule.py
@Author  ：湛蓝
@Date    ：2022/8/21 10:31 
'''
import torch
from torch import nn

from d2l import torch as d2l
from EncoderDecoder import AttentionDecoder, Encoder, Decoder

class Seq2SeqEncoder(Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        #嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        #选择gru来实现编码器
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        #X的形状（batch_size,num_steps,embed_size)
        X = self.embedding(X)
        #变更变换维度
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)
        #因为每条语句之间没有序列关系，所以不需要传入state， 此时rnn参数state默认为None
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state[0]的形状:(num_layers,batch_size,num_hiddens)
        return output, state

class Seq2SeqDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        #编码器最终的隐状态在每一个时间步都作为解码器的输入序列的一部分
        self.embedding = nn.Embedding(vocab_size, embed_size)
        #编码器的最后一步的隐状态用来初始化解码器的隐状态
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)

        self.linear = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        '''
        用encoder的输出作为隐状态， encoder输出的就是最浓缩的信息
        :param enc_outputs: encoder的输出，包括(outputs, state)
        :param args:
        :return:
        '''
        return enc_outputs[1]

    def forward(self, X, state=None):
        #虽然state给的是None但是实际上此处的state不可能是None
        X = self.embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps，然后直接与X拼到一起送入rnn
        #上下文信息就是编码器隐状态的最后
        context = state[-1].repeat(X.shape[0], 1, 1)
        #按照embeding拼接
        X_and_context = torch.cat((X, context), 2)

        output, state = self.rnn(X_and_context, state)
        output = self.linear(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size) 不用展开吗？
        # state[0]的形状:(num_layers,batch_size,num_hiddens)
        return output, state

class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)
        #编码器最终的输出在每一个时间步都作为解码器的输入序列的一部分 即上下文
        #自己编写的加性attention
        self.attention = d2l.AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout)

        self.embedding = nn.Embedding(vocab_size, embed_size)
        #编码器的最后一步的隐状态用来初始化解码器的隐状态
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.linear = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        '''
        用encoder的输出作为隐状态， encoder输出的就是最浓缩的信息
        :param enc_outputs: encoder的输出，包括(outputs, state)
        :param args:
        :return:
        '''
        outputs, hidden_state = enc_outputs
        #output.shape=(num_steps,batch_size,num_hiddens)
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens )

    def forward(self, X, state=None):
        #虽然state给的是None但是实际上此处的state不可能是None
        enc_outputs, hidden_state, enc_valid_lens = state
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:         #X的第一个token是<bos>
            query = torch.unsqueeze(hidden_state[-1], dim=1)    #上个时刻rnn最后的输出
            #enc_outputs.shape = (nums_step, batch_size, wordembed_size)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            #x.shape=(batch_size,embed_size+context)
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention_weights)
            pass
        outputs = self.linear(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]
    @property
    def attention_weights(self):    #给画图用的
        return self._attention_weights


def sequence_mask(X, valid_len, value=0):
    '''
    暂时不清楚咋实现的，考虑自己写这个算法
    :param X:
    :param valid_len:
    :param value:
    :return:
    '''
    myself = True
    if myself:
        for index, v_len in enumerate(valid_len):
            X[index, v_len:] = value
    else:
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
    return X
# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(sequence_mask(X, torch.tensor([1, 2])))

if __name__ == '__main__':
    # encoder = Seq2SeqEncoder(10, 8, 16, 2)
    # encoder.eval()
    # X = torch.zeros((4, 7), dtype=torch.long)
    # output, state = encoder(X)
    # print(output.shape, state.shape)
    #
    # decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
    #                          num_layers=2)
    # decoder.eval()
    # state = decoder.init_state(encoder(X))
    # output, state = decoder(X, state)
    # print(output.shape, state.shape)
    pass
