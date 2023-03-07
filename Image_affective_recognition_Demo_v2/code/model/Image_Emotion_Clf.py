"""
Given an image guess a distribution over the emotion labels.

The MIT License (MIT)
Originally created in 2020, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.notebook import tqdm as tqdm_notebook
from Image_affective_recognition_Demo_v2.code.original_model.neural_models.mlp import MLP
from Image_affective_recognition_Demo_v2.code.original_model.neural_models.resnet_encoder import ResnetEncoder
from Image_affective_recognition_Demo_v2.code.artemis_model.utils.stats import AverageMeter
from Image_affective_recognition_Demo_v2.code.model.Abs_Attr_Model import Abs_Attr_Model


'''
img_encoder = ResnetEncoder('resnet34', adapt_image_size=1).unfreeze(level=7, verbose=True)
    img_emb_dim = img_encoder.embedding_dimension()

    
    abstract_projection_net=None
    abstract_ground_dim=0
    if args.use_abstract:
        abstract_in_dim = 1000
        abstract_ground_dim = 9
        # obviously one could use more complex nets here instead of using a "linear" layer.
        # in my estimate, this is not going to be useful:)
        abstract_projection_net = nn.Sequential(*[nn.Linear(abstract_in_dim, abstract_ground_dim), nn.ReLU()])
        img_emb_dim=img_emb_dim+abstract_ground_dim
    # here we make an MLP closing with LogSoftmax since we want to train this net via KLDivLoss
    clf_head = MLP(img_emb_dim, [100, args.n_emotions], dropout_rate=0.3, b_norm=True, closure=torch.nn.LogSoftmax(dim=-1))
'''

class ImageEmotionClassifier(nn.Module):
    def __init__(self, opt, vocab_size):
        super(ImageEmotionClassifier, self).__init__()
        self.word_clf = Abs_Attr_Model(opt,vocab_size)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 512))
        self.clf_head = MLP(512 , [100, opt.n_emotions], dropout_rate=0.3, b_norm=True, closure=torch.nn.LogSoftmax(dim=-1))

    def __call__(self, img,obj_feat,attr_label,emtion_feat):
        word_loss, attr_index, img_emd = self.word_clf(img,obj_feat,attr_label)
        att_pred = self.word_clf.embed_det(self.word_clf.embed(attr_index))
        final_feat = torch.cat([img_emd,att_pred],dim=1)
        final_feat = self.adaptive_pool(final_feat).squeeze(1)
        logits = self.clf_head(final_feat)
        kld = torch.nn.KLDivLoss(reduction='batchmean').to(logits.device)
        logits = logits.to(torch.float64)
        img_loss = kld(logits, emtion_feat)/len(logits)
        
        loss = word_loss+img_loss
        return loss, attr_index, logits


def single_epoch_train(model, data_loader, optimizer, device):
    epoch_loss = AverageMeter()
    model.train()
    for batch in tqdm.tqdm(data_loader):
        img_feat = batch['image'].to(device)
        obj_feat = batch['obj_feat'].to(device)
        abs_feat = batch['abstract_data'].to(device)
        emtion_feat = batch['emotion_his'].to(device)
        loss, att_pred, logits= model(img_feat,obj_feat,abs_feat,emtion_feat)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.update(loss.item())
    return epoch_loss.avg

def single_epoch_eval(model, data_loader, device,args):
    epoch_loss = AverageMeter()
    precision_a = AverageMeter()
    recall_a = AverageMeter()
    acc_a = AverageMeter()
    model.eval()
    for batch in tqdm.tqdm(data_loader):
        img_feat = batch['image'].to(device)
        obj_feat = batch['obj_feat'].to(device)
        abs_feat = batch['abstract_data'].to(device)
        emtion_feat = batch['emotion_his'].to(device)
        loss, att_pred, logits= model(img_feat,obj_feat,abs_feat,emtion_feat)
        precision,recall = compute_index(abs_feat,att_pred,args.selected_num,data_loader.batch_size)
        acc = evaluate_argmax_prediction(emtion_feat, logits)
        epoch_loss.update(loss.item())
        precision_a.update(precision)
        recall_a.update(recall)
        acc_a.update(acc)
    return epoch_loss.avg,precision_a.avg,recall_a.avg,acc_a.avg
    
def evaluation_index(model, data_loader, device,args):
    precision_sum = AverageMeter()
    recall_sum = AverageMeter()
    acc_sum = AverageMeter()
    
    for batch in tqdm.tqdm(data_loader):
        img_feat = batch['image'].to(device)
        obj_feat = batch['obj_feat'].to(device)
        abs_feat = batch['abstract_data'].to(device)
        emtion_feat = batch['emotion_his'].to(device)
        _, att_pred, logits= model(img_feat,obj_feat,abs_feat,emtion_feat)
        precision,recall = compute_index(abs_feat,att_pred,args.selected_num,data_loader.batch_size)
        acc = evaluate_argmax_prediction(emtion_feat, logits)
        precision_sum.update(precision)
        recall_sum.update(recall)
        acc_sum.update(acc)
    return precision_sum.avg,recall_sum.avg,acc_sum.avg
    
    
def compute_index(att_feat,attr_index,selected_num,batch_size):
    precision=AverageMeter()
    recall=AverageMeter()

    attr_labels = att_feat.cpu().numpy()  
    attr_labels = attr_labels[:,:1000]
    for k in range(len(attr_index)):
        DAS = attr_index[k].cpu().numpy()
        detect_attribute = np.zeros([1,1000])
        detect_attribute[:,DAS] = 1
        TP = np.sum(detect_attribute * attr_labels[k])
        tp_p = TP/selected_num
        tp_r = TP/max(1,np.sum(attr_labels[k]))
        precision.update(tp_p)
        recall.update(tp_r)   
    return precision.avg,recall.avg

def evaluate_argmax_prediction(labels, guesses):
    labels = labels.cpu().numpy()
    guesses = guesses.cpu().detach().numpy()
    unique_max = (labels == labels.max(1, keepdims=True)).sum(1) == 1
    umax_ids = np.where(unique_max)[0]
    gt_max = np.argmax(labels[unique_max], 1)
    max_pred = np.argmax(guesses[umax_ids], 1)
    return (gt_max == max_pred).mean()