from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import os
#import misc.utils as utils
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from collections import OrderedDict
from Abstract_predict_Demo_v2.code.data_object.AverageMeter import AverageMeter

import torch
from torch import nn
from torchvision import models
import json

import tqdm
'''
Abs_Attr_Model:  Abstract_Attribute_model
'''
class Abs_Attr_Model(nn.Module):
    def __init__(self, opt,vocab_size):
        super(Abs_Attr_Model, self).__init__()
        self.vocab_size = vocab_size
        self.selected_num = opt.selected_num
        self.input_encoding_size = opt.input_encoding_size
        self.att_hid_size = opt.att_hid_size
        self.fc_feat_size = opt.fc_feat_size
        
        self.img_embed = models.resnet101(pretrained=True)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embed = torch.nn.Embedding(vocab_size,self.input_encoding_size)
        self.embed_det = nn.Linear(self.input_encoding_size, self.att_hid_size)
        self.feats_det = nn.Linear(self.fc_feat_size, self.att_hid_size)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()
        
        if opt.drop > 0:
            modules = list(self.img_embed.children())
            modules = modules[:-opt.drop]
            self.img_embed = nn.Sequential(*modules)
        
        for p in self.img_embed.parameters():
            p.requires_grad = False

    def init_weight(self):
        for p in self.parameters():
            if (len(p.shape) == 2):
                init.xavier_normal_(p)
            else:
                init.constant_(p,0)
        return
    
    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks
  
    def multimodal_detector(self,att_feats, attr_labels,selected_num):
        sig = self.sigmoid 
   
        attr_emd = self.embed_det(self.embed(torch.Tensor([range(0,self.vocab_size)]).long().cuda()).detach()).squeeze(0)  #1000*512 '0' is the start/end token
        feats_emd = self.feats_det(att_feats)        #bs*max*512
        
        #compute the similarity
        b_attr =  attr_emd.t().unsqueeze(0).expand(feats_emd.shape[0],attr_emd.shape[1],attr_emd.shape[0])
        logits = torch.bmm(feats_emd.unsqueeze(1),b_attr) #bs*max*1000
        p_raw = torch.log(1.0 - sig(logits)+1e-7)
        
        #merge the probability
        p_merge = torch.sum(p_raw,dim=1,keepdim=False) #bs*1000
        p_final = 1.0 - torch.exp(p_merge)
        p_final = torch.clamp(p_final,0.01,0.99)
        #print(p_final)
        top_prob,attr_index = torch.topk(p_final,selected_num,dim=1)
        
        if(attr_labels is not None and attr_labels.shape[0] == p_final.shape[0]):
        #if(attr_labels is not None):
            alpha_factor = torch.tensor(0.95).cuda()
            gamma = torch.tensor(2.0).cuda()
            alpha_factor = torch.where(torch.eq(attr_labels, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(attr_labels, 1.), 1. - p_final, p_final)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
             
            bce = -(attr_labels * torch.log(p_final) + (1.0 - attr_labels) * torch.log(1.0 - p_final))
             
            cls_loss = focal_weight * bce
             
            focal_loss = torch.sum(cls_loss)/torch.max(torch.tensor(1.0).cuda(),torch.sum(attr_labels))
        else:
            focal_loss = torch.tensor(0.0).cuda()
        

        self.p_det = p_final
        
        return focal_loss,attr_index
    
    def forward(self, img_feats,  attr_feats):
        img_feats = self.img_embed(img_feats)
        img_feats = self.adaptive_pool(img_feats)
        img_feats = img_feats.permute(0, 2, 3, 1)      # bring feature-dim last.
        img_feats = torch.squeeze(torch.squeeze(img_feats, 1), 1) 
        focal_loss,attr_index = self.multimodal_detector(img_feats, attr_feats, self.selected_num)
        
        return focal_loss, attr_index


def single_epoch_train(model, data_loader, optimizer, device):
    epoch_loss = AverageMeter()
    model.train()
    for batch in tqdm.tqdm(data_loader):
        img_feat = batch['image'].to(device)
        abs_feat = batch['abstract_data'].to(device)
        focal_loss, _ = model(img_feat,abs_feat)

        # Back prop.
        optimizer.zero_grad()
        focal_loss.backward()
        optimizer.step()
        b_size = len(abs_feat)
        epoch_loss.update(focal_loss.item(), b_size)
    return epoch_loss.avg

def single_epoch_eval(model, data_loader, device,args):
    epoch_loss = AverageMeter()
    precision_a = AverageMeter()
    recall_a = AverageMeter()
    model.eval()
    for batch in tqdm.tqdm(data_loader):
        img_feat = batch['image'].to(device)
        abs_feat = batch['abstract_data'].to(device)
        focal_loss, attr_index = model(img_feat,abs_feat)
        precision,recall = compute_index(abs_feat,attr_index,args.selected_num,data_loader.batch_size)
        epoch_loss.update(focal_loss.item(), len(batch))
        precision_a.update(precision,len(batch))
        recall_a.update(recall, len(batch))
    return epoch_loss.avg,precision_a.avg,recall_a.avg
    
def evaluation_index(model, data_loader, device,args):
    precision_sum = 0
    recall_sum = 0
    
    for batch in tqdm.tqdm(data_loader):
        img_feat = batch['image'].to(device)
        abs_feat = batch['abstract_data'].to(device)
        _, attr_index = model(img_feat,abs_feat)
        precision,recall = compute_index(abs_feat,attr_index,args.selected_num,len(batch))
        precision_sum=precision_sum+precision
        recall_sum=recall_sum+recall
    return precision_sum/len(data_loader),recall_sum/len(data_loader)
    
    
def compute_index(att_feat,attr_index,selected_num,batch_size):
    precision=0
    recall=0

    attr_labels = att_feat.cpu().numpy()  

    for k in range(len(attr_index)):
        DAS = attr_index[k].cpu().numpy()
        detect_attribute = np.zeros([1,1000])
        detect_attribute[:,DAS] = 1
        TP = np.sum(detect_attribute * attr_labels[k][:,:1000])
        tp_p = TP/selected_num
        tp_r = TP/max(1,np.sum(attr_labels[k][:,:1000]))
        precision += tp_p
        recall += tp_r   
    return precision/batch_size,recall/batch_size