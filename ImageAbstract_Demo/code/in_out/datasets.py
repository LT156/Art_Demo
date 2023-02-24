"""
The MIT License (MIT)
Originally in 2020, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

from tkinter import X
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from artemis.evaluation.emotion_alignment import image_to_emotion, wscnet_image_to_emotion
from artemis.emotions import emotion_to_int
from collections import Counter


class AffectiveCaptionDataset(Dataset):
    """ Basically, an image, with a caption, and an indicated emotion.
    """
    def __init__(self,image_files,abstract_words_feature, tokens, emotions,img_emo_id, n_emotions=9, img_transform=None, one_hot_emo=True):
        super(AffectiveCaptionDataset, self).__init__()
        self.unique_id = img_emo_id
        self.image_files = image_files
        self.tokens = tokens
        self.emotions = emotions
        self.n_emotions = n_emotions
        self.img_transform = img_transform
        self.one_hot_emo = one_hot_emo
        self.abstract_words_feature = abstract_words_feature
       

    def __getitem__(self, index):
        text = np.array(self.tokens[index]).astype(dtype=np.long)
        abstract_data = np.array(self.abstract_words_feature[index]).astype(dtype=np.long)

        if self.image_files is not None:
            img = Image.open(self.image_files[index])

            if img.mode is not 'RGB':
                img = img.convert('RGB')

            if self.img_transform is not None:
                img = self.img_transform(img)
        else:
            img = []

        if self.n_emotions > 0:
            if self.one_hot_emo:
                emotion = np.zeros(self.n_emotions, dtype=np.float32)
                emotion[self.emotions[index]] = 1
            else:
                emotion = self.emotions[index]
        else:
            emotion = []
        unique_id = self.unique_id[index]
        res = {'unique_id':unique_id,'image': img,'abstract_data':abstract_data,'emotion_index' :self.emotions[index] , 'emotion': emotion,'tokens': text, 'index': index}
        return res

    def __len__(self):
        return len(self.tokens)


class WordsPredictDataset(Dataset):
    def __init__(self, image_files,  abstract_features=None,img_transform=None, rgb_only=True):
        super(WordsPredictDataset, self).__init__()
        self.image_files = image_files
        self.abstract_features = abstract_features
        self.img_transform = img_transform
        self.rgb_only = rgb_only

    def __getitem__(self, index):
        img = Image.open(self.image_files[index])
        abstract_data = np.array(self.abstract_features[index]).astype(dtype=np.float32)

        if self.rgb_only and img.mode is not 'RGB':
            img = img.convert('RGB')

        if self.img_transform is not None:
            img = self.img_transform(img)
        res = {'image': img,'abstract_data':abstract_data, 'index': index}
        return res

    def __len__(self):
        return len(self.image_files)

class AbstrstWordsDataset(Dataset):
    def __init__(self, image_files,  emotion,abstract_features=None,img_transform=None, rgb_only=True):
        super(AbstrstWordsDataset, self).__init__()
        self.image_files = image_files
        self.emotions = emotion
        self.abstract_features = abstract_features
        self.img_transform = img_transform
        self.rgb_only = rgb_only

    def __getitem__(self, index):
        img = Image.open(self.image_files[index])
        abstract_data = np.array(self.abstract_features[index]).astype(dtype=np.float32)

        if self.rgb_only and img.mode is not 'RGB':
            img = img.convert('RGB')

        if self.img_transform is not None:
            img = self.img_transform(img)
            

        emotion = self.emotions[index]
        res = {'image': img, 'emotion': emotion,'abstract_data':abstract_data, 'index': index}
        return res

    def __len__(self):
        return len(self.image_files)

