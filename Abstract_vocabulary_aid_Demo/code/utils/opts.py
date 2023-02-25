from __future__ import print_function
import os
import pathlib
import argparse
import argparse
import json
import pprint
import os.path as osp
from datetime import datetime
from RAIVS.artemis_model.in_out.basics import create_dir

demo_path='F:/work/Image_emotion_analysis/artemis-master'
def str2bool(v):
    """ boolean values for argparse
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_train_speaker_arguments(notebook_options=None, save_args=False):

    parser = argparse.ArgumentParser(description='training-a-neural-speaker')

    # config
    parser.add_argument('--cfg', type=str, default=demo_path+'/Abstract_vocabulary_aid_Demo/config/configs_local/rl_train_small.yml',
                    help='configuration; similar to what is used in detectron')
    parser.add_argument(
        '--set_cfgs', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]/n This has higher priority'
             'than cfg file but lower than other args. (You can only overwrite'
             'arguments that have alerady been defined in config file.)',
        default=[], nargs='+')

    ## Non-optional arguments
    parser.add_argument('-log-dir', type=str,  default='',help='where to save training-progress, model, etc.')
    parser.add_argument('-data-dir', type=str,   default='',help='path to ArtEmis/COCO preprocessed data')
    parser.add_argument('-img-dir', type=str,   default='',help='path to top image (e.g., WikiArt) dir')
    parser.add_argument('-references_pkl_file', type=str,  default='')
    parser.add_argument('-beam-size', type=int,  default=5,help='where to save training-progress, model, etc.')
    parser.add_argument('-sampling-config-file', type=str,   default='',help='path to ArtEmis/COCO preprocessed data')
    # Model parameters
    parser.add_argument('--img-dim', type=int, default=256, help='images will be resized to be squared with this many pixels')
    parser.add_argument('--lanczos', type=str2bool, default=True, help='apply lanczos resampling when resizing')
    parser.add_argument('--atn-spatial-img-size', type=int, help='optional, if provided reshapes the spatial output dimension of the '
                                                                 'visual encode in this X this "pixels" using average-pooling. ')
    parser.add_argument('--atn-cover-img-alpha', type=float, default=1, help='attention to cover entire image when '
                                                                             'marginalized over tokens')
    parser.add_argument('--attention-dim', type=int, default=512)
    parser.add_argument('--rnn-hidden-dim', type=int, default=512)
    parser.add_argument('--word-embedding-dim', type=int, default=128)
    parser.add_argument('--vis-encoder', type=str, default='resnet34', choices=['resnet18',
                                                                                'resnet34',
                                                                                'resnet50',
                                                                                'resnet101'], help='visual-encoder backbone')
    parser.add_argument('--dropout-rate', type=float, default=0.1)
    parser.add_argument('--teacher-forcing-ratio',  type=int, default=1)

    # Emotion Label
    parser.add_argument('--use-emo-grounding', type=str2bool, default=True)
    parser.add_argument('--emo-grounding-dims', nargs=2, type=int, default=[9, 9], help='[input] number of emotions x the'
                                                                                        'the size of the projection layer that '
                                                                                        'will be used to transform the one-hot emotion'
                                                                                        'to a grounding vector.')


    # Training parameters
    parser.add_argument('--resume-path', type=str, help='model-path to resume from')
    parser.add_argument('--fine-tune-data', type=str)
    parser.add_argument('--batch-size', type=int, default=128)#128
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--encoder-lr', type=float, default=1e-4)
    parser.add_argument('--decoder-lr', type=float, default=5e-4)
    parser.add_argument('--max-train-epochs', type=int, default=50)
    parser.add_argument('--train-patience', type=int, default=5, help='maximum consecutive epochs where the validation '
                                                                      'Neg-LL does not improve before we stop training.')
    parser.add_argument('--lr-patience', type=int, default=2, help='maximum waiting of epochs where the validation '
                                                                   'Neg-LL does not improve before we reduce the'
                                                                   'learning-rate.')
    parser.add_argument('--save-each-epoch', type=str2bool, default=True, help='Save the model at each epoch, else will only save'
                                                                               'the one that achieved the minimal '
                                                                               'Negative-Log-Likelihood in the validation split.')
    #RL Training parameters
    parser.add_argument('-gt_json', type=str, default='', help='ground truth file for reward computing')
    parser.add_argument('-cached_tokens', type=str, default='',help='n-grams for ciderD to compute tf-idf')
    parser.add_argument('--sprcial_word', type=str, default='',help='notional word to fix reward,eg: V, N, Adj')
    parser.add_argument('-use_scst_training', type=str2bool, default=True, help='flag to use scst model')
    parser.add_argument('-img2emo_checkpoint', type=str, default='', help='predict img to emo')
    parser.add_argument('-text2emo_path', type=str, default='', help='predict text to emo')
    parser.add_argument('-model_ckp', type=str, default='', help='model after xe train')

    # Misc
    parser.add_argument('--dataset', type=str, default='artemis')
    parser.add_argument('--random-seed', type=int, default=2021)
    parser.add_argument('--debug', default=False, type=str2bool)
    parser.add_argument('--use-timestamp', default=True, type=str2bool)

    # Parse arguments
    if notebook_options is not None:  # Pass options directly
        args = parser.parse_args(notebook_options)
    else:
        args = parser.parse_args() # Read from command line.

    #load params from yml
    # step 1: read cfg_fn
    args = parser.parse_args()
    if args.cfg is not None or args.set_cfgs is not None:
        from RAIVS.utils.config import CfgNode
        if args.cfg is not None:
            cn = CfgNode(CfgNode.load_yaml_with_base(args.cfg))
        else:
            cn = CfgNode()
        if args.set_cfgs is not None:
            cn.merge_from_list(args.set_cfgs)
        for k,v in cn.items():
            if not hasattr(args, k):
                print('Warning: key %s not in args' %k)
            setattr(args, k, v)
        args = parser.parse_args(namespace=args)
    
    # Check if args are valid
    #assert args.rnn_size > 0, "rnn_size should be greater than 0"
    #assert args.num_layers > 0, "num_layers should be greater than 0"
    if args.use_timestamp:
        timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        args.log_dir = create_dir(osp.join(args.log_dir, timestamp))

    # pprint them
    args_string = pprint.pformat(vars(args))
    print(args_string)

    if save_args:
        out = osp.join(args.log_dir, 'config.json.txt')
        with open(out, 'w') as f_out:
            json.dump(vars(args), f_out, indent=4, sort_keys=True)

    return args

def parse_test_speaker_arguments(notebook_options=None):
    """ Parameters for testing (sampling) a neural-speaker.
    :param notebook_options: list, if you are using this via a jupyter notebook
    :return: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description='testing-a-neural-speaker')

    # config
    parser.add_argument('--cfg', type=str, default=demo_path+'/Abstract_vocabulary_aid_Demo/config/configs_local/rl_decode_test_small.yml',
                    help='configuration; similar to what is used in detectron')
    parser.add_argument(
        '--set_cfgs', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]/n This has higher priority'
             'than cfg file but lower than other args. (You can only overwrite'
             'arguments that have alerady been defined in config file.)',
        default=[], nargs='+')
   
    ## Basic required arguments
    parser.add_argument('-speaker-saved-args', type=str,  default='',help='config.json.txt file for saved speaker model (output of train_speaker.py)')
    parser.add_argument('-speaker-checkpoint', type=str,  default='', help='saved model checkpoint ("best_model.pt" (output of train_speaker.py)')
    parser.add_argument('-img-dir', type=str,  default='', help='path to top image dir (typically that\'s the WikiArt top-dir)')
    parser.add_argument('-out-file', type=str,  default='',help='file to save the sampled utterances, their attention etc. as a pkl')
    ## Basic optional arguments
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'val', 'rest'], help='set the split of the dataset you want to annotate '
                                                                                                            'the code will load the dataset based on the dir-location marked '
                                                                                                            'in the input config.json.txt file. ' 
                                                                                                            'this param has no effect if a custom-data-csv is passed.')

    parser.add_argument('--custom-data-csv', type=str, help='if you want to annotate your own set of images. Please '
                                                            'see the code for what this csv should look like. ')

    parser.add_argument('--subsample-data', type=int, default=-1, help='if not -1, will subsample the underlying dataset'
                                                                        'and will annotated only this many images.')


    ## Optional arguments controlling the generation/sampling process
    parser.add_argument('--max-utterance-len', type=int, help='maximum allowed lenght for any sampled utterances. If not given '
                                                              'the maximum found in the underlying dataset split will be used.'
                                                              'Fot the official ArtEmis split for deep-nets that is 30 tokens.')

    parser.add_argument('--drop-unk', type=str2bool, default=True, help='if True, do not create samples that contain the '
                                                                        'unknown token')

    parser.add_argument('--drop-bigrams', type=str2bool, default=True, help='if True, prevent the same bigram to occur '
                                                                            'twice in a sampled utterance')


    parser.add_argument('--sampling-config-file', type=str, help='Note. if max-len, drop-unk '
                                                                 'and drop-bigrams are not specified in the json'
                                                                 'the directly provided values of these parameters '
                                                                 'will be used.')


    parser.add_argument('--random-seed', type=int, default=2021, help='if -1 it won\'t have an effect; else the sampler '
                                                                      'becomes deterministic')

    parser.add_argument('--img2emo-checkpoint', type=str, default='',help='checkpoint file of image-2-emotion classifier that will '
                                                               'be used to sample the grounding emotion that will be used '
                                                               'by the speaker, if you pass an emotionally-grouned speaker. '
                                                               'Note. if you pass/use an emo-grounded speaker - this argument '
                                                               'becomes required, except if you are using your own custom-data-csv '
                                                               'where you can specify the grounding emotion manually.' )

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--n-workers', type=int)

    parser.add_argument('--compute-nll', type=str2bool, default=False, help='Compute the negative-log-likelihood of '
                                                                            'the dataset under the the saved speaker model.')



    # Parse arguments
    if notebook_options is not None:  # Pass options directly
        args = parser.parse_args(notebook_options)
    else:
        args = parser.parse_args() # Read from command line.

    #load params from yml
    # step 1: read cfg_fn
    args = parser.parse_args()
    if args.cfg is not None or args.set_cfgs is not None:
        from RAIVS.utils.config import CfgNode
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

def parse_test_evaluation_arguments(notebook_options=None):
    parser = argparse.ArgumentParser(description='evaluation_mycode of synthetic captions')
    
    # config
    parser.add_argument('--cfg', type=str, default=demo_path+'/Abstract_vocabulary_aid_Demo/config/configs_local/rl_evaluation_small.yml',
                    help='configuration; similar to what is used in detectron')
    parser.add_argument(
        '--set_cfgs', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]/n This has higher priority'
             'than cfg file but lower than other args. (You can only overwrite'
             'arguments that have alerady been defined in config file.)',
        default=[], nargs='+')

    parser.add_argument('-cached_tokens', type=str,  default='corpus')
    parser.add_argument('-references-pkl-file', type=str )
    parser.add_argument('-captions-pkl-file', type=str )
    parser.add_argument('-text2emo-path', type=str,   help='path to neural-net')
    parser.add_argument('-vocab-path', type=str,   help='vobab file path')
    parser.add_argument('-vocab-path-14469', type=str,   help='vobab file path')
    parser.add_argument('-save-file', type=str)
    parser.add_argument('--gpu-id', type=str, default='0')
    parser.add_argument('--mask-file', type=str, help='constrain the captions/ground-truth')
    parser.add_argument('--train-bert-embeddings', type=str)
    parser.add_argument('--lcs-sample-size', type=int, nargs=2, default=[5000, 500])
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--mode', type=str, default='test')

    args = parser.parse_args()

    if args.cfg is not None or args.set_cfgs is not None:
        from RAIVS.utils.config import CfgNode
        if args.cfg is not None:
            cn = CfgNode(CfgNode.load_yaml_with_base(args.cfg))
        else:
            cn = CfgNode()
        if args.set_cfgs is not None:
            cn.merge_from_list(args.set_cfgs)
        for k,v in cn.items():
            if not hasattr(args, k):
                print('Warning: key %s not in args' %k)
            setattr(args, k, v)
        args = parser.parse_args(namespace=args)

    args_string = pprint.pformat(vars(args))
    print(args_string)

    if not os.path.exists(args.save_file):
        os.makedirs(args.save_file)
    return args


if __name__=='__main__':
    args=parse_train_speaker_arguments()
    print(args)
