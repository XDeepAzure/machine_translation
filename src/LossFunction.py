#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：machine_translation 
@File    ：LossFunction.py
@Author  ：湛蓝
@Date    ：2022/8/22 22:37 
'''
import torch
from torch import nn, Tensor


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):

    def forward(self, input: Tensor, target: Tensor, valid_len) -> Tensor:
        weights = torch.ones_like(target)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        #要求预测结果放到第二个维度上
        unweighted_loss = super().forward(input.permute(0, 2, 1), target.to(torch.long))
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


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

if __name__ == '__main__':
    loss = MaskedSoftmaxCELoss()
    l = loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long),
         torch.tensor([4, 2, 0]))
    print(l)
    pass
