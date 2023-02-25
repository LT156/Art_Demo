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
from RAIVS.feat_extracted_model.ResNetWSL import ResNetWSL
from RAIVS.feat_extracted_model.ClassWisePool import ClassWisePool


def describe_model(vocab, args):
    """ Describe the architecture of a SAT speaker with a resnet encoder.
    :param vocab:
    :param args:
    :return:
    """
    word_embedding = nn.Embedding(len(vocab), args.word_embedding_dim, padding_idx=vocab.pad)

    encoder = ResnetEncoder(args.vis_encoder, adapt_image_size=args.atn_spatial_img_size).unfreeze()
    encoder_out_dim = encoder.embedding_dimension()

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
    e_encoder = ResnetEncoder(args.vis_encoder, adapt_image_size=args.atn_spatial_img_size).unfreeze()
    # Initialize the model for this run
    # model_ft, input_size = initialize_model(args.num_classes, args.num_maps, args.feature_extract, use_pretrained=True)
    model = nn.ModuleDict({'encoder': encoder, 'decoder': decoder})
    return model



def initialize_model(num_classes, num_maps, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    """ Resnet101
    """
    model_ft =ResnetEncoder('resnet34').unfreeze(level=7, verbose=True).resnet    
    pooling = nn.Sequential()
    pooling.add_module('class_wise', ClassWisePool(num_maps))
    pooling2 = nn.Sequential()
    pooling2.add_module('class_wise', ClassWisePool(num_classes))
    model_ft = ResNetWSL(model_ft, num_classes, num_maps, pooling, pooling2)
    
    input_size = 256
    checkpoint = 'F:/work/Image_emotion_analysis/WSCNet_master/out_emo/version_2/wscnet.pt'
    pre = torch.load(checkpoint)
    print('情感检测加载完毕')
    model_ft.load_state_dict(pre['model_state_dict'])
    set_parameter_requires_grad(model_ft)
    
    return model_ft, input_size

def set_parameter_requires_grad(model):
    # if feature_extracting:
    for param in model.parameters():
        param.requires_grad = False
        

    