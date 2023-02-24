"""
Given an image guess a distribution over the emotion labels.

The MIT License (MIT)
Originally created in 2020, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import torch
import torch.nn.functional as F
from torch import nn
import tqdm

from artemis.utils.stats import AverageMeter


class ImageWordsClassifier(nn.Module):
    def __init__(self, img_encoder, clf_head):
        super(ImageWordsClassifier, self).__init__()
        self.img_encoder = img_encoder
        self.clf_head = clf_head

    def __call__(self, img):
        feat = self.img_encoder(img)
        logits = self.clf_head(feat)
        return logits

class ImageWordsEmotionClassifier(nn.Module):
    def __init__(self, img_encoder, img2word_clf, clf_head):
        super(ImageWordsEmotionClassifier, self).__init__()
        self.img_encoder = img_encoder
        self.img2word_clf = img2word_clf
        self.word_net = nn.Linear(1000,9)
        self.clf_head = clf_head
        self.softmax = nn.Softmax(dim=-1)
        
        for p in self.img2word_clf.parameters():
                p.requires_grad = False

    def __call__(self, img):
        img_feat = self.img_encoder(img)
        word_pred = self.img2word_clf(img,None)
        word_feat = self.word_net(word_pred)
        word_feat = torch.cat([img_feat,word_feat],dim=1)
        logits = self.clf_head(word_feat)
        word_pred = self.softmax(word_pred)
        logits = self.softmax(logits)
        return word_pred,logits

def single_epoch_train(model, data_loader, criterion, optimizer, device,args=None):
    epoch_loss = AverageMeter()
    model.train()
    for batch in tqdm.tqdm(data_loader):
        img = batch['image'].to(device)
        emotion = batch['emotion'].to(device)
        abstract_data = batch['abstract_data'].to(device)
        word_pred,logits = model(img)

        # Calculate loss
        loss1 = criterion(word_pred, abstract_data)
        loss2 = criterion(logits, emotion)
        loss = loss1+loss2
        # Back prop.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        b_size = len(abstract_data)
        epoch_loss.update(loss.item(), b_size)
    return epoch_loss.avg


@torch.no_grad()
def evaluate_on_dataset(model, data_loader, criterion, device, detailed=True, kl_div=True,args=None):
    epoch_loss = AverageMeter()
    model.eval()
    epoch_confidence = []
    for batch in tqdm.tqdm(data_loader):
        img = batch['image'].to(device)
        emotion = batch['emotion'].to(device)
        abstract_data = batch['abstract_data'].to(device)
        word_pred,logits = model(img)

        # Calculate loss
        loss1 = criterion(word_pred, abstract_data)
        loss2 = criterion(logits, emotion)
        loss = loss1+loss2

        if detailed:
            if kl_div:
                epoch_confidence.append(torch.exp(logits).cpu())  # logits are log-soft-max
            else:
                epoch_confidence.append(F.softmax(logits, dim=-1).cpu()) # logits are pure logits

        b_size = len(abstract_data)
        epoch_loss.update(loss.item(), b_size)

    if detailed:
        epoch_confidence = torch.cat(epoch_confidence).numpy()

    return epoch_loss.avg, epoch_confidence