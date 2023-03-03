import os
import pprint
import torch
import pickle
import argparse
import pandas as pd
import numpy as np
from Abstract_predict_Demo_v2.code.utils.config import CfgNode
from Abstract_predict_Demo_v2.code.artemis_model.in_out.neural_net_oriented import  torch_save_model
from Abstract_predict_Demo_v2.code.artemis_model.in_out.basics import create_dir
from Abstract_predict_Demo_v2.code.model.Abs_Attr_Model import Abs_Attr_Model, evaluation_index, single_epoch_eval, single_epoch_train
from Abstract_predict_Demo_v2.code.in_out.dataloader_to_pack import dataloader_pack


def init_model(args):
    '''
    args:vocab_size\input_encoding_size\att_hid_size
    '''
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    model = Abs_Attr_Model(args,len(vocab))
    return  model

def train(model, data_loaders, optimizer, device,args):
    no_improvement = 0
    min_eval_loss = np.Inf
    best_model=None

    for epoch in range(1, args.max_train_epochs+1):
        train_loss = single_epoch_train(model, data_loaders['train'], optimizer, device)
        print('Epoch {} Train Loss: {:.6f}'.format(epoch,train_loss))

        eval_loss,precision,recall = single_epoch_eval(model, data_loaders['val'], device, args)
        print('Epoch {} Eval Loss: {:.6f}\t Eval_Precision:{:.4f}\t Eval_Recall: {:.4f} '.format(epoch,eval_loss,precision,recall))

        if eval_loss < min_eval_loss:
            min_eval_loss = eval_loss
            no_improvement = 0
            print('Epoch {}. Validation loss improved!'.format(epoch))
            torch_save_model(model, args.checkpoint_file)
       
            precision,recall = evaluation_index(model, data_loaders['test'], device,args)
            print('Epoch {} Test_Precision:{:.4f}\t Test_Recall: {:.4f} '.format(epoch,precision,recall))
            best_model = model
        else:
            no_improvement += 1
            print('Epoch {} not improved'.format(epoch))  
        
        if no_improvement >=5 :
            print('Breaking at epoch {}. Since for 5 epoch we observed no (validation) improvement.'.format(epoch))
            break
    return best_model
    
def get_args():
    parser = argparse.ArgumentParser(description='testing-image2emoclf')
    # config
    parser.add_argument('--cfg', type=str, default='F:/work/Image_emotion_analysis/artemis-master/Abstract_predict_Demo_v2/config/image2word_local.yml',
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
    if os.path.exists(args.checkpoint_file)==False:
        create_dir(args.checkpoint_file)
    
    ## Prepare the artemis dataset (merge it with the emotion-histograms.)
    ## Prepare the artemis dataset (merge it with the emotion-histograms.)
    loaded = np.load(args.additional_data_file,allow_pickle=True)
    feature_df= pd.DataFrame(loaded['feature_df'],columns=loaded['columns'])
    artemis_data = feature_df

    ## prepare data
    data_loaders, datasets = dataloader_pack(artemis_data, args)
    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    
    model = init_model(args).to(device)
    optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': 5e-4}])
    
    print('Starting Training……')
    model = train(model, data_loaders, optimizer, device,args)
    precision,recall = evaluation_index(model, data_loaders['test'], device,args)
    print('Finially, Test_Precision:{:.4f}\t Test_Recall: {:.4f} '.format(precision,recall))
    print('END')