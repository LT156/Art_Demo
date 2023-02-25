"""
Decoding module for a neural speaker (with attention capabilities).

The MIT License (MIT)
Originally created at 06/15/19, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""
import numpy as np
import torch
import time
import torch.nn.functional as F
from RAIVS.reinforcement_learning.reward import get_reward
from RAIVS.model.xe_train_and_val import describe_decode
import tqdm

from RAIVS.artemis_model.utils.stats import AverageMeter

def beam_search_train(mode, epoch,model, data_loader, device, optimizer,tb_writer=None,**kwargs):
    """
    :param model (encoder, decoder)
    :param data_loader:
    :param beam_size:
    :param drop_unk:
    :return:

        hypotheses_alphas: list carrying the attention maps over the encoded-pixel space for each produced token.
    Note: batch size must be one.
    """
    print_freq = 10
    beam_size = kwargs.get('beam_size', 5)  
    kwargs['batch_size'] = data_loader.batch_size
    kwargs['vocab'] = model.decoder.vocab
    decode_settings={'temperature':1,'drop_unk':False,'drop_bigrams':False,'remove_bad_endings':False}
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    steps_taken = (epoch-1) * len(data_loader.dataset)
    model.train()
    decoder= model.decoder

    running_loss=0
    running_reward=0
    running_reward_baseline=0
    idx=0
    for batch in tqdm.tqdm(data_loader):
        start = time.time() 
        batch_size=len(batch['image'])
        unique_id = batch['unique_id']
        image = batch['image'].to(device)  # (b_s, 3, H, W)
        captions = batch['tokens'].to(device)
        data_time.update(time.time() - start)
        
        if decoder.uses_aux_data:
            emotion = batch['emotion'].to(device)
            aux_feat=decoder.auxiliary_net(emotion)
           
        else:
            emotion=None
            aux_feat=None

        res=describe_decode('scst',mode,model, image, aux_feat, captions, decode_settings)
        _, log_probs,_=res
        reward,reward_baseline,fixed_reward=get_reward(unique_id,emotion,res,captions,**kwargs)
        
        loss=compute_rl_loss(log_probs,reward,reward_baseline,fixed_reward)
        loss= loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        running_reward += reward.mean().item()
        running_reward_baseline += reward_baseline.mean().item()

        batch_time.update(time.time() - start)
        steps_taken += batch_size
        # Print status
        if print_freq is not None and idx % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'reward_loss {loss:.50f} \t'
                    'reward {reward:.20f} \t'
                    'reward_baseline {reward_baseline:.20f}'.format(epoch, idx, len(data_loader),
                                                                            batch_time=batch_time,
                                                                                data_time=data_time,
                                                                                loss=loss.item(),
                                                                                reward=reward.mean().item(),
                                                                                reward_baseline=reward_baseline.mean().item()))
        idx=idx+1
    if tb_writer is not None:
        tb_writer.add_scalar('training-entropy-loss-with-batch-granularity', running_reward_baseline, steps_taken)


    loss = running_loss / len(data_loader)
    reward = running_reward / len(data_loader)
    reward_baseline = running_reward_baseline / len(data_loader)
    return loss,reward,reward_baseline



@torch.no_grad()
def beam_search_val(mode, model, data_loader, device,**kwargs):
    """
    :param model (encoder, decoder)
    :param data_loader:
    :param beam_size:
    :param drop_unk:
    :return:

        hypotheses_alphas: list carrying the attention maps over the encoded-pixel space for each produced token.
    Note: batch size must be one.
    """

    beam_size = kwargs.get('beam_size', 5)  
    decode_settings={'temperature':0.3,'drop_unk':True,'drop_bigrams':True,'remove_bad_endings':True}
    kwargs['batch_size'] = data_loader.batch_size
    kwargs['vocab'] = model.decoder.vocab

    model.eval()
    #decoder = model.caption_decoder.attentive_decoder
    decoder = model.decoder
    aux_feat = None

    running_loss=0
    running_reward=0
    running_reward_baseline=0
    #for each batch
    for batch in tqdm.tqdm(data_loader):  
        batch_size=len(batch['image'])
        image = batch['image'].to(device)  # (b_s, 3, H, W)
        unique_id = batch['unique_id']
        captions = batch['tokens'].to(device)

        aux_feat=None
        aux_data=None
        if decoder.uses_aux_data:
            aux_data=batch['emotion'].to(device)
            aux_feat = decoder.auxiliary_net(aux_data)
        else:
            aux_feat=None
        with torch.no_grad():
            res=describe_decode('scst',mode, model,image, aux_feat, captions, decode_settings)
        _, log_probs,_=res
        reward,reward_baseline,fixed_reward=get_reward(unique_id,aux_data,res,captions,**kwargs)
        
        loss = compute_rl_loss(log_probs,reward,reward_baseline,fixed_reward)
        loss= loss

        running_loss += loss.item()
        running_reward += reward.mean().item()
        running_reward_baseline += reward_baseline.mean().item()

    loss = running_loss / len(data_loader)
    reward = running_reward / len(data_loader)
    reward_baseline = running_reward_baseline / len(data_loader)
    return loss,reward,reward_baseline

def compute_rl_loss(log_probs,reward,reward_baseline,fixed_reward):
    device=log_probs.device
    mask=(log_probs!=0).int()
    max_length=mask.sum(-1)
    
    reward_final=(torch.from_numpy(reward - reward_baseline).to(device)/max_length).unsqueeze(-1)
    fixed_reward=torch.from_numpy(fixed_reward).to(device)
    reward=reward_final*(1+fixed_reward)*mask
    loss=torch.mean(-log_probs* reward, -1)
    loss = loss.mean()
    return loss
    
def beam_sampler(mode, model,data_loader,beam_size,device):
    model.eval()
    decoder = model.decoder
    vocab = model.decoder.vocab

    decode_settings={'temperature':0.3,'drop_unk':True,'drop_bigrams':True,'remove_bad_endings':True}

    caption_str=[]
    alphas_list=[]
    #for each batch
    for batch in tqdm.tqdm(data_loader):  
        image = batch['image'].to(device)  # (b_s, 3, H, W)
        captions = batch['tokens'].to(device)
        aux_feat=None
        abstract_data=None
        if decoder.uses_aux_data:
            aux_data=batch['emotion'].to(device)
            aux_feat = decoder.auxiliary_net(aux_data)
        if decoder.uses_abstract_data:
            abstract_data=batch['abstract_data'].to(device)
            abstract_data = decoder.abstract_projection_net(abstract_data)
            
        with torch.no_grad():
            res=describe_decode('scst',mode, model,image, aux_feat,abstract_data, captions, decode_settings)
        caption_beams_out, _,alpha_beams_out =res
        caption=[" ".join([vocab.idx2word[int(idx)] for idx in scent  if idx not in [0,1,2]]) for scent in caption_beams_out[:,0,:]]
        alpha_=alpha_beams_out[:,0,:,:].squeeze(1)
        alphas_=[]
        for i in range(len(caption)):
            c_len=len(caption[i].split(" "))
            alpha=alpha_[i][:c_len+2]
            alphas_.append(alpha)
        alphas_list=alphas_list+alphas_
        caption_str=caption_str+caption
    return caption_str,alphas_list  

