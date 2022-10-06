#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：machine_translation 
@File    ：device.py
@Author  ：湛蓝
@Date    ：2022/8/23 10:48 
'''
import torch


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists.

    Defined in :numref:`sec_use_gpu`"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

if __name__ == '__main__':
    pass
