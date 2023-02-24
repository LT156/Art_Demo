import os
import pprint
import torch
import argparse
import pandas as pd
import os.path as osp
import numpy as np
from torch import nn as nn

from ImageAbstract_Demo.code.in_out.neural_net_oriented import torch_load_model, torch_save_model, save_state_dicts
from ImageAbstract_Demo.code.in_out.neural_net_oriented import image_emotion_distribution_df_to_pytorch_dataset
from ImageAbstract_Demo.code.model.image_emotion_clf import ImageWordsEmotionClassifier,ImageWordsClassifier
from ImageAbstract_Demo.code.model.image_emotion_clf import single_epoch_train, evaluate_on_dataset

from ImageAbstract_Demo.code.artemis_model.in_out.basics import create_dir
from ImageAbstract_Demo.code.original_model.neural_models.mlp import MLP
from ImageAbstract_Demo.code.original_model.neural_models.resnet_encoder import ResnetEncoder





## helper function.
## to evaluate how well the model does according to the class that it finds most likely
## note it only concerns the predictions on examples (images) with a single -unique maximizer- emotion
def evaluate_argmax_prediction(dataset, guesses):
    labels = dataset.emotions
    labels = np.vstack(labels.to_numpy())
    unique_max = (labels == labels.max(1, keepdims=True)).sum(1) == 1
    umax_ids = np.where(unique_max)[0]
    gt_max = np.argmax(labels[unique_max], 1)
    max_pred = np.argmax(guesses[umax_ids], 1)
    return (gt_max == max_pred).mean()

def init_model(args):
    img_encoder = ResnetEncoder('resnet34', adapt_image_size=1).unfreeze(level=7, verbose=True)
    img_emb_dim = img_encoder.embedding_dimension()+9
    clf_head = MLP(img_emb_dim, [100, args.n_emotions], dropout_rate=0.3, b_norm=True, closure=torch.nn.LogSoftmax(dim=-1))

    img2word_clf = ImageWordsClassifier(None,None)
    img2word_clf = torch_load_model(args.img2word_clf)   
    model = ImageWordsEmotionClassifier(img_encoder,img2word_clf, clf_head);
    return  model

def test_and_plot(args,data_loaders,criterion,device):
    model = torch_load_model(args.checkpoint_file)    
    _, test_confidence = evaluate_on_dataset(model, data_loaders['test'], criterion, device, detailed=True,args=args)

    # How well it does on test images that have strong majority in emotions?
    labels = data_loaders['test'].dataset.emotions
    labels = np.vstack(labels.to_numpy())

    for use_strong_domi in [True, False]:
        print('use_strong_domi:', use_strong_domi)
        if use_strong_domi:
            dominant_max = (labels.max(1) > 0.5)
        else:
            dominant_max = (labels.max(1) >= 0.5)

        umax_ids = np.where(dominant_max)[0]
        gt_max = np.argmax(labels[dominant_max], 1)
        max_pred = np.argmax(test_confidence[umax_ids], 1)    

        print('Test images with dominant majority', dominant_max.mean())
        print('Guess-correctly', (gt_max == max_pred).mean(), '\n')
    dataset = data_loaders['test'].dataset        
    arg_max_acc = evaluate_argmax_prediction(dataset, test_confidence)
    print('有且仅有最大值情感的准确度: {:.3f}'.format(arg_max_acc))

def get_args():
    parser = argparse.ArgumentParser(description='testing-image2emoclf')
    # config
    parser.add_argument('--cfg', type=str, default='F:/work/Image_emotion_analysis/artemis-master/ImageAbstract_Demo/config/evaluation_local.yml',
                    help='configuration; similar to what is used in detectron')
    parser.add_argument(
        '--set_cfgs', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]\n This has higher priority'
             'than cfg file but lower than other args. (You can only overwrite'
             'arguments that have alerady been defined in config file.)',
        default=[], nargs='+')
    #load params from yml
    # step 1: read cfg_fn
    args = parser.parse_args()
    if args.cfg is not None or args.set_cfgs is not None:
        from ImageAbstract_Demo.code.utils.config import CfgNode
        if args.cfg is not None:
            cn = CfgNode(CfgNode.load_yaml_with_base(args.cfg))
        else:
            cn = CfgNode()
        if args.set_cfgs is not None:
            cn.merge_from_list(args.set_cfgs)
        for k,v in cn.items():
            setattr(args, k, v)
        args = parser.parse_args(namespace=args)

    # pprint them
    print('\nParameters Specified:')
    args_string = pprint.pformat(vars(args))
    print(args_string)
    print('\n')
    return args
    


if __name__=='__main__':

    args = get_args()
    checkpoint_file = args.checkpoint_file
    
    loaded = np.load(args.df_data_dir,allow_pickle=True)
    feature_df= pd.DataFrame(loaded['feature_df'],columns=loaded['columns'])
    artemis_data = feature_df
    ## prepare data
    data_loaders, datasets = image_emotion_distribution_df_to_pytorch_dataset(artemis_data, args)

    ## Prepate the Neural-Net Stuff (model, optimizer etc.)
    ## This is what I used for the paper with minimal hyper-param-tuning. You can use different nets/configs here...
    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.KLDivLoss(reduction='batchmean').to(device)
    model = init_model(args).to(device)
    test_and_plot(args,data_loaders,criterion,device)
    print('END')