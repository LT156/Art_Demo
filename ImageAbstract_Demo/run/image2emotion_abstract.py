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

def train(model, data_loaders, criterion, optimizer, device,args):
    # set to True, if you are not using a pretrained model
    max_train_epochs = 25
    no_improvement = 0
    min_eval_loss = np.Inf
    best_model=None

    for epoch in range(1, max_train_epochs+1):
        train_loss = single_epoch_train(model, data_loaders['train'], criterion, optimizer, device,args=args)
        print('Train Loss: {:.3f}'.format(train_loss))

        eval_loss, _ = evaluate_on_dataset(model, data_loaders['val'], criterion, device, detailed=False,args=args)
        print('Eval Loss: {:.3f}'.format(eval_loss))

        if eval_loss < min_eval_loss:
            min_eval_loss = eval_loss
            no_improvement = 0
            print('Epoch {}. Validation loss improved!'.format(epoch))
            torch_save_model(model, checkpoint_file)
                
            test_loss, test_confidence = evaluate_on_dataset(model, data_loaders['test'], criterion, device, detailed=True,args=args)
            print('Test Loss: {:.3f}'.format(test_loss))                

            dataset = data_loaders['test'].dataset        
            arg_max_acc = evaluate_argmax_prediction(dataset, test_confidence)
            print('Test arg_max_acc: {:.3f}'.format(arg_max_acc))
            best_model = model
        else:
            no_improvement += 1
            print('Epoch {}. Validation loss Not improved!'.format(epoch))
        
        if no_improvement >=5 :
            print('Breaking at epoch {}. Since for 5 epoch we observed no (validation) improvement.'.format(epoch))
            break
    return best_model

def get_args():
    parser = argparse.ArgumentParser(description='testing-image2emoclf')
    # config
    parser.add_argument('--cfg', type=str, default='F:/work/Image_emotion_analysis/artemis-master/ImageAbstract_Demo/config/image2emotion_local.yml',
                    help='configuration; similar to what is used in detectron')
    parser.add_argument(
        '--set_cfgs', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]/n This has higher priority'
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
    print('/nParameters Specified:')
    args_string = pprint.pformat(vars(args))
    print(args_string)
    print('/n')
    return args
    


if __name__=='__main__':

    args = get_args()
    if os.path.exists(args.save_dir):
        create_dir(args.save_dir)
    checkpoint_file = osp.join(args.save_dir, 'best_model.pt')
    # minor parameters
    GPU_ID = 0 
    ## Prepare the artemis dataset (merge it with the emotion-histograms.)
    loaded = np.load(args.df_data_dir,allow_pickle=True)
    feature_df= pd.DataFrame(loaded['feature_df'],columns=loaded['columns'])
    artemis_data = feature_df
    print('Image numbers:',len(artemis_data))
    artemis_data=artemis_data[:200]
    
    ## prepare data
    data_loaders, datasets = image_emotion_distribution_df_to_pytorch_dataset(artemis_data, args)

    ## Prepate the Neural-Net Stuff (model, optimizer etc.)
    ## This is what I used for the paper with minimal hyper-param-tuning. You can use different nets/configs here...
    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.KLDivLoss(reduction='batchmean').to(device)
    model = init_model(args).to(device)
    optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': 5e-4}])
    
    print('Starting Training……')
    model = train(model, data_loaders, criterion, optimizer, device,args)
    print('END')