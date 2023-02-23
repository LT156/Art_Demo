import os
import pprint
import torch
import argparse
import pandas as pd
import os.path as osp
import numpy as np
from ast import literal_eval

from Abstract_predict_Demo.code.artemis_model.in_out.neural_net_oriented import torch_load_model, torch_save_model
from Abstract_predict_Demo.code.artemis_model.in_out.neural_net_oriented import image_emotion_distribution_df_to_pytorch_dataset
from Abstract_predict_Demo.code.artemis_model.in_out.basics import create_dir

from Abstract_predict_Demo.code.original_model.neural_models.mlp import MLP
from Abstract_predict_Demo.code.original_model.neural_models.resnet_encoder import ResnetEncoder
from Abstract_predict_Demo.code.original_model.neural_models.image_emotion_clf import ImageWordsClassifier
from Abstract_predict_Demo.code.original_model.neural_models.image_emotion_clf import single_epoch_train, evaluate_on_dataset

def evaluate_argmax_prediction(dataset, guesses):
    abstract_data = dataset.abstract_features
    abstract_data = np.vstack(abstract_data.to_numpy())
    unique_max = (abstract_data == abstract_data.max(1, keepdims=True)).sum(1) == 1
    umax_ids = np.where(unique_max)[0]
    gt_max = np.argmax(abstract_data[unique_max], 1)
    max_pred = np.argmax(guesses[umax_ids], 1)
    return (gt_max == max_pred).mean()




def test_and_plot(args,data_loaders,criterion,device):
    model = torch_load_model(args.checkpoint_file)    
    test_loss, test_confidence = evaluate_on_dataset(model, data_loaders['test'], criterion, device, detailed=True)

    dataset = data_loaders['test'].dataset        
    arg_max_acc = evaluate_argmax_prediction(dataset, test_confidence)
    print('top1 准确率: {:.3f}'.format(arg_max_acc))
    
    abstract_features= np.vstack(dataset.abstract_features.to_numpy())
    preds = test_confidence
    dominant_max = abstract_features.max(1)>0
    umax_ids = np.where(dominant_max)[0]
    
    gt_domin = abstract_features[umax_ids]
    preds_domin = preds[umax_ids] 
    
    
    gt_domin = torch.from_numpy(gt_domin)
    preds_domin = torch.from_numpy(preds_domin)
    topk = torch.count_nonzero(gt_domin, dim=1).reshape(-1, 1)
    
    sources = []
    for i in range(topk.shape[0]):
        ground_topk_index=gt_domin.topk(int(topk[i])).indices.reshape(-1,)
        preds_topk_index = preds_domin.topk(int(topk[i])).indices.reshape(-1,)
        if set(ground_topk_index.tolist())==set(preds_topk_index.tolist()):
            sources.append(1)
        else:
            sources.append(0)
    print('抽象词汇预测的准确率：',np.mean(sources))
    print('END')

    
def get_args():
    parser = argparse.ArgumentParser(description='testing-image2emoclf')
    # config
    parser.add_argument('--cfg', type=str, default='F:/work/Image_emotion_analysis/artemis-master/Abstract_predict_Demo/config/word_evaluation_local.yml',
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
        from Abstract_predict_Demo.code.utils.config import CfgNode
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
    checkpoint_file = args.checkpoint_file
    # minor parameters
    GPU_ID = 0 
    
    ## Prepare the artemis dataset (merge it with the emotion-histograms.)
     ## Prepare the artemis dataset (merge it with the emotion-histograms.)
    loaded = np.load(args.df_data_dir,allow_pickle=True)
    feature_df= pd.DataFrame(loaded['feature_df'],columns=loaded['columns'])
    artemis_data = feature_df[:200]

    ## prepare data
    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    data_loaders, datasets = image_emotion_distribution_df_to_pytorch_dataset(artemis_data, args)
    test_and_plot(args,data_loaders,criterion,device)
    print('done')