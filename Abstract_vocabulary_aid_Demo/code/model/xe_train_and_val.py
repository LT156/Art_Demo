"""
Decoding module for a neural speaker (with attention capabilities).

The MIT License (MIT)
Originally created at 06/15/19, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""
import torch
import random
import time
import warnings
import tqdm
import math
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils import clip_grad_norm_

from artemis.utils.stats import AverageMeter
from Abstract_vocabulary_aid_Demo.code.model.describe_decode import describe_decode

bad_endings=['are','am','a','and','an','after','it','is','like','in',
            'the','to','before','her','his','just','not','near','very',
            'of','on','for','with','upon','at','what','looking','feel',
            'i','me','he','she','they','s','their','his','her','than',
            'about','but','that','happen']


def single_epoch_train(train_loader, model, criterion, optimizer, epoch, device,  tb_writer=None, **kwargs):
    """ Perform training for one epoch.
    :param train_loader: DataLoader for training data
    :param model: nn.ModuleDict with 'encoder', 'decoder' keys
    :param criterion: loss layer
    :param optimizer: optimizer
    :param epoch: epoch number
    :param device:
    """
    alpha_c = kwargs.get('alpha_c', 1.0)  # Weight of doubly stochastic (attention) regularization.
    grad_clip = kwargs.get('grad_clip', 5.0) # Gradient clipping (norm magnitude)
    print_freq = kwargs.get('print_freq', 100)
    use_emotion = kwargs.get('use_emotion', False)
    use_abstract = kwargs.get('use_abstract', False)
    mode = kwargs.get('mode', 'original_model')
    

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    entropy_loss_meter = AverageMeter()  # entropy loss (per word decoded)
    total_loss_meter = AverageMeter()
    start = time.time()
    steps_taken = (epoch-1) * len(train_loader.dataset)
    model.train()

    for i, batch in enumerate(train_loader):
        imgs = batch['image'].to(device)
        
        caps_token = batch['tokens'].to(device)
        b_size = len(imgs)
        data_time.update(time.time() - start)
        
        emotion=None
        if use_emotion:
            emotion = batch['emotion'].to(device)
        abstract_data=None
        if use_abstract:
            abstract_data = batch['abstract_data'].to(device)
        res = describe_decode('xe', mode, model, imgs, emotion, abstract_data,caps_token,None)
        
        logits, caps_sorted, decode_lengths, alphas, sort_ind = res

        # Since we decoded starting with <sos>, the targets are all words after <sos>, up to <eos>
        targets = caps_sorted[:, 1:]

        # Remove time-steps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        logits = pack_padded_sequence(logits, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        #ent_loss = criterion(logits.data, targets.data)
        data1 = logits.data
        data2 = targets.data.to(torch.int64)
        # one_hot = torch.zeros(np.array(batch_size, num_class, device=torch.device('cuda:0')).scatter_(1, label, 1)
        # https://blog.csdn.net/qi_sama/article/details/122402880
        ent_loss = criterion(data1, data2)
        # Calculation of loss
        total_loss = ent_loss

        # Add doubly stochastic attention regularization
        # Note. some implementation simply do this like: d_atn_loss = alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        # here we take care of the fact that some samples in the same batch have more/less tokens than others.
        if alpha_c > 0:
            total_energy = torch.from_numpy(np.array(decode_lengths)) / alphas.shape[-1]   # n_tokens / num_pixels
            total_energy.unsqueeze_(-1)  # B x 1
            total_energy = total_energy.to(device)
            d_atn_loss = alpha_c * ((total_energy - alphas.sum(dim=1)) ** 2).mean()
            total_loss += d_atn_loss

        # Back prop.
        optimizer.zero_grad()
        total_loss.backward()
        if grad_clip is not None:
            clip_grad_norm_(model.parameters(), grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        entropy_loss_meter.update(ent_loss.item(), sum(decode_lengths))
        total_loss_meter.update(total_loss.item(), sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()
        steps_taken += b_size

        # Print status
        if print_freq is not None and i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader),
                                                                            batch_time=batch_time,
                                                                            data_time=data_time,
                                                                            loss=total_loss_meter))
        if tb_writer is not None:
            tb_writer.add_scalar('training-entropy-loss-with-batch-granularity', entropy_loss_meter.avg, steps_taken)

    return total_loss_meter.avg

@torch.no_grad()
def negative_log_likelihood(model, data_loader, device,args):
    """
    :param model:
    :param data_loader:
    :param device:
    :param phase:
    :return:
    """
    model.eval()
    nll = AverageMeter()
    for batch in data_loader:
        imgs = batch['image'].to(device)
        caps_token = batch['tokens'].to(device)
        
        # TODO Refactor
        emotion = None 
        if model.decoder.uses_aux_data:
            emotion = batch['emotion'].to(device)
        abstract_data = None
        if model.decoder.uses_abstract_data==True:
            abstract_data = batch['abstract_data'].to(device)
        res = describe_decode('xe', args.mode, model, imgs, emotion,abstract_data, caps_token,None)
       
        
        logits, caps_sorted, decode_lengths, alphas, sort_ind = res
        #logits, caps_sorted, decode_lengths, alphas, sort_ind = model(imgs,caps_token, emotion,len(imgs))
        # Since we decoded starting with <sos>, the targets are all words after <sos>, up to <eos>
        targets = caps_sorted[:, 1:]

        # Remove time-steps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        logits = pack_padded_sequence(logits, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        #loss = F.cross_entropy(logits.data, targets.data)
        data1 = logits.data
        data2 = targets.data.to(torch.int64)
        # one_hot = torch.zeros(np.array(batch_size, num_class, device=torch.device('cuda:0')).scatter_(1, label, 1)
        # https://blog.csdn.net/qi_sama/article/details/122402880
        ent_loss = F.cross_entropy(data1, data2)
        # Calculation of loss
        loss = ent_loss
        nll.update(loss.item(), sum(decode_lengths))
    return nll.avg

