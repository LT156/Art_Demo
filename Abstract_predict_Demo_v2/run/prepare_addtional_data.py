
import numpy as np
import pandas as pd
import pprint
import argparse
import pickle
from ast import literal_eval
from Abstract_predict_Demo_v2.code.data_object.Vocabulary import Vocabulary


def get_args():
    parser = argparse.ArgumentParser(description='prepare_additional_data')
    # config
    parser.add_argument('--cfg', type=str, default='F:/work/Image_emotion_analysis/artemis-master/Abstract_predict_Demo_v2/config/prepared_data_local2.yml',
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
    '''
    1.读取token
    2.读取vocab
    3.获得每个文本的属性集1000维
    4.存起来
    '''
    args = get_args()
    
    loaded = np.load(args.text_file,allow_pickle=True)
    df= pd.DataFrame(loaded['df'],columns=loaded['columns'])[:5000]
    vocab = Vocabulary.load(args.vocab_file)
   
    
    painting = []
    abs_feature = []
    split = []
    for key,g in df.groupby('painting'):
        painting.append(key)
    
        abs_feat = np.zeros([len(vocab)])
        for  text in g['tokens_lem_encoded']:
            for w_id in text:
                if w_id<1000 and w_id>=0:
                    abs_feat[w_id] += 1
        abs_feat = (abs_feat>0).astype(np.float32) 
        abs_feature.append(abs_feat)
        split.append(g['split'].iloc[0])
    abs_feat_df = pd.DataFrame({'painting':painting,'abs_feature':abs_feature,'split':split})
    
    img_emo_df = pd.read_csv(args.image_hists_file)
    result_df = pd.merge(img_emo_df,abs_feat_df,on='painting',how='inner')
    np.savez_compressed(args.out_file, feature_df=result_df,columns=result_df.columns)
    
    print('END')
    
    