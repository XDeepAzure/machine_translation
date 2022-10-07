#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：machine_translation 
@File    ：train.py
@Author  ：湛蓝
@Date    ：2022/8/23 10:46 
'''
import datetime
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from EncoderDecoder import EncoderDecoder
from LossFunction import MaskedSoftmaxCELoss
from NetModule import Seq2SeqAttentionDecoder, Seq2SeqEncoder, Seq2SeqDecoder
from dataset import translationDataset
from device import try_gpu
from predict import predict, bleu

def train_epoch(net, train_loader,loss, optimizer, tgt_vocab, device,epoch):
    cross_loss, total_tokens, i = 0, 0, 0
    for t in train_loader:
        i += 1
        print("第%d轮，%d/%d" % (epoch, i, len(train_loader)))
        optimizer.zero_grad()
        X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in t]

        #bos形状(1,batch_size)->(batch_size,1)
        bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                           device=device).reshape(-1, 1)
        # 强制教学  Y的形状(batch_size, num_steps)
        dec_input = torch.cat([bos, Y[:, :-1]], 1)      #在num_steps后面拼接
        Y_hat, _ = net(X, dec_input, X_valid_len)
        l = loss(Y_hat, Y, Y_valid_len)
        l.sum().backward()
        grad_clipping(net, 10)  #这里梯度剪裁放大一点
        num_tokens = Y_valid_len.sum()
        optimizer.step()
        with torch.no_grad():
            cross_loss += l.sum()
            total_tokens += num_tokens
    return cross_loss, total_tokens

def train(net:nn.Module, train_loader, lr, num_epochs, num_steps, src_vocab, tgt_vocab, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    writer = SummaryWriter()
    net.train()
    for epoch in range(num_epochs):
        start = time.process_time()
        cross_loss, total_tokens = train_epoch(net, train_loader,loss,
                                               optimizer,tgt_vocab,device, epoch)
        writer.add_scalar('loss_attention', cross_loss, epoch)
        writer.add_text('translation','i\'m home .=>' +\
             predict(net, 'i\'m home .', src_vocab, tgt_vocab, num_steps, device)[0], epoch)
        print(f'loss {cross_loss / total_tokens:.3f}, {total_tokens / (time.process_time()-start) :.1f} '
              f'tokens/sec on {str(device)}')

    # save_checkpoint(net, optimizer, loss)

def save_checkpoint(net, optimizer, loss):
    dic = {'model':net, 'optimizer': optimizer, 'loss_fn': loss}
    torch.save(dic, 'checkpoint-%s.pth' % datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
    pass

def grad_clipping(net, theta):
    '''梯度剪裁'''
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad==True]
    else:
        params = net.params
    norm =  torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


if __name__ == '__main__':
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps, lr, num_epochs = 64, 10, 0.005, 400
    device = try_gpu()
    train_dataset = translationDataset(num_steps, batch_size, device, num_examples=2000)
    src_vocab, tgt_vocab = train_dataset.src_vocab, train_dataset.tgt_vocab
    train_loader = DataLoader(train_dataset, batch_size)

    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    # decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    
    decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)

    net = EncoderDecoder(encoder, decoder)

    # net = torch.load('./checkpoint-220826-145906.pth')['model']

    train(net, train_loader, lr, num_epochs, num_steps, src_vocab, tgt_vocab, device)
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, attention_weight_seq = predict(
            net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
    pass
