#!/usr/bin/env python
# coding: utf-8

import os
import torch
import pandas as pd
import itertools
from tqdm.notebook import tqdm
from artemis.utils.vocabulary import Vocabulary
from artemis.in_out.basics import unpickle_data, pickle_data
from artemis.in_out.neural_net_oriented import torch_load_model
from artemis.evaluation_fixed.single_caption_per_image import apply_basic_evaluations
from Abstract_vocabulary_aid_Demo.code.utils.opts import parse_test_evaluation_arguments

def list_txt(path, list=None):
    if list != None:
        file = open(path, 'w')
        file.write(str(list))
        file.close()
        return None
    else:
        file = open(path, 'r')
        rdlist = eval(file.read())
        file.close()
        return rdlist

if __name__ == '__main__':
    args=parse_test_evaluation_arguments()
    """
    "cuda:" + str(args.gpu_id) if torch.cuda.is_available() else 
    """
    device = torch.device("cpu")
    vocab = Vocabulary.load(args.vocab_path_14469)
    txt2emo_clf  = torch_load_model(args.text2emo_path, map_location=device)
    txt2emo_clf = txt2emo_clf.to(device)

    caption_data = next(unpickle_data(args.captions_pkl_file))
    gt_data = next(unpickle_data(args.references_pkl_file))
    gt_data = gt_data[args.split]

    if args.mask_file:
        mask = next(unpickle_data(args.mask_file))
        print('Using a mask to keep {} of data'.format(mask.mean().round(4)))
    else:
        mask = pd.Series([True] * len(gt_data))

    if args.debug:
        print('***Debugging***')
        gt_data = gt_data.iloc[0:100]
        args.lcs_sample_size = [10, 2]

    test_utters = gt_data['references_pre_vocab']
    test_utters = list(itertools.chain(*test_utters))  # undo the grouping per artwork to a single large list
    print('Training Utterances', len(test_utters))
    unique_test_utters = set(test_utters)
    print('Unique Training Utterances', len(unique_test_utters))

    results = []
    methods_to_do = {'bleu', 'cider', 'ciderD','spice', 'meteor', 'rouge', 'emo_alignment'}
    for config_i, captions_i,_ in tqdm(caption_data):
        if args.debug:
            captions_i = captions_i.iloc[0:100]

        merged = pd.merge(captions_i,gt_data,on='painting',how='inner')  # this ensures proper order of captions to gt (via accessing merged.captions)
        merged = merged[mask].reset_index(drop=True)

        hypothesis = merged.caption
        references = merged.references
        ref_emotions = merged.emotion
        if len(results) == 0:
            print('|Masked Captions| Size:', len(hypothesis))

        default_lcs_sample = [25000, 800]
        basic_eval_res = apply_basic_evaluations(hypothesis, references, ref_emotions, txt2emo_clf, vocab,args.cached_tokens,
                                    lcs_sample=default_lcs_sample,
                                    train_utterances=unique_test_utters,
                                    device=device,
                                    methods_to_do=methods_to_do)
        
        eval_res = basic_eval_res  # + fancy_eval_res
        results.append([config_i, pd.DataFrame(eval_res)])

        if args.debug:
            if len(results) == 2:
                break
    if not os.path.exists(args.save_file):
        os.mkdir(args.save_file)
    list_txt(args.save_file+'/result.txt', results)
    pickle_data(args.save_file+'/result.pkl', results)


