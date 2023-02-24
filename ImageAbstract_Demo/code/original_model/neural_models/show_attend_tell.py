"""
A custom implementation of Show-Attend-&-Tell for ArtEmis: Affective Language for Visual Art

The MIT License (MIT)
Originally created in early 2020, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""
import torch
from torch import nn
from .resnet_encoder import ResnetEncoder
from .attentive_decoder import AttentiveDecoder
from artemis.neural_models.image_emotion_clf import ImageEmotionClassifier
from artemis.neural_models.mlp import MLP
from artemis.in_out.neural_net_oriented import torch_load_model
from RAIVS.feat_extracted_model.ClassWisePool import ClassWisePool
from RAIVS.feat_extracted_model.ResNetWSL import ResNetWSL



def describe_model(vocab, args):
    """ Describe the architecture of a SAT speaker with a resnet encoder.
    :param vocab:
    :param args:
    :return:
    """
    # 特征模块
    # Initialize the model for this run
    model_ft, input_size = initialize_model(args.num_classes, args.num_maps, args.feature_extract, use_pretrained=True)
    
    # 编码器模块
    word_embedding = nn.Embedding(len(vocab), args.word_embedding_dim, padding_idx=vocab.pad)
    
    encoder = ResnetEncoder(args.vis_encoder, adapt_image_size=args.atn_spatial_img_size).unfreeze()
    encoder_out_dim = encoder.embedding_dimension()
    if args.mode=='cat':
        encoder_out_dim = encoder_out_dim+1
    # 解码器模块
    emo_ground_dim = 0
    emo_projection_net = None
    if args.use_emo_grounding:
        emo_in_dim = args.emo_grounding_dims[0]
        emo_ground_dim = args.emo_grounding_dims[1]
        # obviously one could use more complex nets here instead of using a "linear" layer.
        # in my estimate, this is not going to be useful:)
        emo_projection_net = nn.Sequential(*[nn.Linear(emo_in_dim, emo_ground_dim), nn.ReLU()])

    decoder = AttentiveDecoder(word_embedding,
                               args.rnn_hidden_dim,
                               encoder_out_dim,
                               args.attention_dim,
                               vocab,
                               dropout_rate=args.dropout_rate,
                               teacher_forcing_ratio=args.teacher_forcing_ratio,
                               auxiliary_net=emo_projection_net,
                               auxiliary_dim=emo_ground_dim)

    model = describe_model(model_ft,encoder,decoder)
    return model


def initialize_model(num_classes, num_maps, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    """ Resnet101
    """
    model_ft =ResnetEncoder('resnet34').unfreeze(level=7, verbose=True).resnet
    

    set_parameter_requires_grad(model_ft, feature_extract)

    pooling = nn.Sequential()
    pooling.add_module('class_wise', ClassWisePool(num_maps))
    pooling2 = nn.Sequential()
    pooling2.add_module('class_wise', ClassWisePool(num_classes))
    model_ft = ResNetWSL(model_ft, num_classes, num_maps, pooling, pooling2)
    
    input_size = 256
    return model_ft, input_size

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class describe_model(nn.Module):
    def __init__(self, emotion_decoder, img_feat, caption_decoder):
       self.emotion_decoder = emotion_decoder
       self.img_feat = img_feat
       self.caption_decoder = caption_decoder
       
    def forward(self, mode, image, label):
        encoder=None
        if mode=='original_model':
            encoder=self.img_feat(image)
        elif mode=='EMOdetected_model':
            img_feat=self.img_feat(image)
            _,_,emo_area=self.emotion_decoder(image)
            feature_map_split_new = []
            for id,index in enumerate(label):
                feature_map_split_new.append(emo_area[id,index,:,:].unsqueeze(0).unsqueeze(0))
            feature_map_split_new = torch.cat(feature_map_split_new,dim=0)

            output, _ = encoder_fuse(img_feat, feature_map_split_new, mode)
            
        output=self.caption_decoder(encoder)
        return output
    
    
def encoder_fuse(encoder_out, feature_map_split, mode):
    batch_size = encoder_out.size(0)
    encoder_dim = encoder_out.size(-1)

    # Flatten image
    encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
    feature_map_split = feature_map_split.permute(0, 2, 3, 1).reshape(batch_size, -1, 1)  # (batch_size, num_pixels, encoder_dim)
    if mode=='cat':
        import torch
        encoder_out=torch.cat([encoder_out,feature_map_split],dim=-1)# b_cX64X513
        encoder_dim=encoder_dim+feature_map_split.size(-1)
    elif mode=='add':
        encoder_out=encoder_out+feature_map_split
    elif mode=='add_cat':
        encoder_out = encoder_out+feature_map_split
        encoder_out=torch.cat([encoder_out,feature_map_split],dim=-1)# b_cX64X513
        encoder_dim=encoder_dim+feature_map_split.size(-1)
    return encoder_out,encoder_dim
