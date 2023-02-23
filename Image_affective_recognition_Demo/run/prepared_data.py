import os
import pprint
import argparse
import pandas as pd
import numpy as np
import os.path as osp

from ast import literal_eval
from torch import nn as nn
from Image_affective_recognition_Demo.code.artemis_model.in_out.basics import create_dir

def normalization(x):
    if sum(x)!=0:
        return (np.array(x) / float(sum(x)))
    else:
        return x
                   

def get_preparedData(image_hists,artemis_data,feature_df):
   
    image_hists.emotion_histogram = image_hists.emotion_histogram.apply(literal_eval)
    image_hists.emotion_histogram = image_hists.emotion_histogram.apply(lambda x: (np.array(x) / float(sum(x))).astype('float32'))
    print(f'Histograms corresponding to {len(image_hists)} images')

    ## keep each image once.
    artemis_data = artemis_data.drop_duplicates(subset=['art_style', 'painting'])
    artemis_data.reset_index(inplace=True, drop=True)

    # keep only relevant info + merge
    artemis_data = artemis_data[['art_style', 'painting', 'split']] 
    artemis_data = artemis_data.merge(image_hists)
    artemis_data = artemis_data.rename(columns={'emotion_histogram': 'emotion_distribution'})
    artemis_data = pd.merge(artemis_data,feature_df,on='painting',how='inner')
    artemis_data.abstract_features = artemis_data.abstract_features.apply(lambda x: (normalization(x)).astype('float32'))
    n_emotions = len(image_hists.emotion_histogram[0])
    print('Using {} emotion-classes.'.format(n_emotions))
    assert all(image_hists.emotion_histogram.apply(len) == n_emotions)
    return artemis_data

def get_args():
    parser = argparse.ArgumentParser(description='testing-image2emoclf')
    # config
    parser.add_argument('--cfg', type=str, default='F:/work/Image_emotion_analysis/artemis-master/Image_affective_recognition_Demo/config/prepared_data_local.yml',
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
        from Image_affective_recognition_Demo.code.utils.config import CfgNode
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
    if os.path.exists(args.save_dir):
        create_dir(args.save_dir)
 

    ## Prepare the artemis dataset (merge it with the emotion-histograms.)
    artemis_data = pd.read_csv(args.artemis_preprocessed_dir)
    image_hists = pd.read_csv(args.image_hists_file)
    loaded = np.load(args.abstract_feature_file,allow_pickle=True)
    feature_df= pd.DataFrame(loaded['feature_df'],columns=loaded['columns'])
    feature_df=feature_df[['painting','abstract_features']]
    print('Annotations loaded:', len(artemis_data))

    artemis_data = get_preparedData(image_hists,artemis_data,feature_df)
    
    np.savez_compressed(args.save_dir, feature_df=artemis_data,columns=artemis_data.columns)
    print('END')