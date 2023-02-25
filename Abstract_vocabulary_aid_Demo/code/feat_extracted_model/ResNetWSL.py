from __future__ import print_function 
from __future__ import division

import torch
import torch.nn as nn
from artemis.neural_models.mlp import MLP

class ResNetWSL(nn.Module):
    
    def __init__(self, model, num_classes, num_maps, pooling, pooling2):
        super(ResNetWSL, self).__init__()
        self.features = model
        img_emb_dim=512
        self.num_ftrs = img_emb_dim

        self.downconv = nn.Sequential(
            nn.Conv2d(img_emb_dim, num_classes*num_maps, kernel_size=1, stride=1, padding=0, bias=True))
        
        self.GAP = nn.AvgPool2d(8)
        self.GMP = nn.MaxPool2d(8)
        self.spatial_pooling = pooling
        self.spatial_pooling2 = pooling2
        self.classifier = nn.Sequential(
            nn.Linear(img_emb_dim*2, num_classes)
            )

    def forward(self, x):
        x = self.features(x)
        x_ori = x  
        # detect branch检测分支
        x = self.downconv(x) 
        x_conv = x              
        #x = self.GAP(x)  #
        x = self.GMP(x)       
        x = self.spatial_pooling(x) 
        x = x.view(x.size(0), -1)
        # cls branch分类分支
        x_conv = self.spatial_pooling(x_conv) 
        x_conv = x_conv * x.view(x.size(0),x.size(1),1,1) 
        feature_map_split=x_conv
        x_conv = self.spatial_pooling2(x_conv) 
        feature_map=x_conv
        x_conv_copy = x_conv
        for num in range(0,512-1):            
            x_conv_copy = torch.cat((x_conv_copy, x_conv),1) 
        x_conv_copy = torch.mul(x_conv_copy,x_ori)
        x_conv_copy = torch.cat((x_ori,x_conv_copy),1) 
        x_conv_copy = self.GAP(x_conv_copy)
        x_conv_copy = x_conv_copy.view(x_conv_copy.size(0),-1)
        x_conv_copy = self.classifier(x_conv_copy)

        x = torch.softmax(x, dim=-1)
        x_conv_copy = torch.softmax(x_conv_copy, dim=-1)
        return x, x_conv_copy,feature_map_split

