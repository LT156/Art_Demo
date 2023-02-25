#!/usr/bin/env python
# coding: utf-8

"""
Load a trained speaker and images/data to create (sample) captions for them.

The MIT License (MIT)
Originally created at 10/3/20, for Python 3.x
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import os
from Abstract_vocabulary_aid_Demo.code.reinforcement_learning.beam_decoder_train_val import beam_sampler
import torch
import json

from Abstract_vocabulary_aid_Demo.code.artemis_model.in_out.basics import pickle_data
from Abstract_vocabulary_aid_Demo.code.utils.opts import parse_test_speaker_arguments
from Abstract_vocabulary_aid_Demo.code.artemis_model.in_out.neural_net_oriented import torch_load_model, load_saved_speaker
from Abstract_vocabulary_aid_Demo.code.artemis_model.captioning.sample_captions import captions_as_dataframe
from Abstract_vocabulary_aid_Demo.code.artemis_model.in_out.datasets import default_grounding_dataset_from_affective_loader
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    '''
    读取额外信息：load_saved_speaker
    
    '''
    args = parse_test_speaker_arguments()
    
    if not os.path.exists(args.out_file):
        os.makedirs(args.out_file)
    use_custom_dataset = False
    if args.custom_data_csv is not None:
        use_custom_dataset = True
    
    with open(args.sampling_config_file) as fin:
        sampling_configs = json.load(fin)



    print('Loaded {} sampling configurations to try.'.format(len(sampling_configs)))
    optional_params = ['max_utterance_len', 'drop_unk', 'drop_bigrams']  # if you did not specify them in the sampling-config
                                                                         # those from the argparse will be used
    

    # Load pretrained speaker & its corresponding train-val-test data. If you do not provide a
    # custom set of images to annotate. Then based on the -split you designated it will annotate this data.
    print('Load data and model')
    speaker, epoch, data_loaders = load_saved_speaker(args.speaker_saved_args, args.speaker_checkpoint,
                                                      with_data=True, verbose=True)
    """
    
    """
    device = torch.device("cuda:" + args.gpu)
    speaker = speaker.to(device)
    eos = speaker.decoder.vocab.eos
    working_data_loader = data_loaders[args.split]

    img2emo_clf = None
    if args.img2emo_checkpoint:
        if args.img2emo_checkpoint == 'original':
            img2emo_clf = 'original'
        elif args.img2emo_checkpoint == 'wscnet':
            img2emo_clf = 'wscnet'
        else:
            img2emo_clf = torch_load_model(args.img2emo_checkpoint, map_location=device)
        
    print('Load data and model')
    data_loader=default_grounding_dataset_from_affective_loader(speaker, working_data_loader, img2emo_clf,device, args.n_workers)
    
    final_results = []
    final_results_1 = []
    for config in sampling_configs:
        for param in optional_params:
            if param not in config:
                config[param] = args.__getattribute__(param)
        print('Sampling with configuration: ', config)

        caption_str,alpha_beams_out=beam_sampler(args.mode, speaker, data_loader,config['beam_size'], device)
        df = captions_as_dataframe(data_loader.dataset, caption_str, wiki_art_data=not use_custom_dataset)
        if not os.path.exists(args.out_file):
            os.mkdir(args.out_file)
        df.to_csv(os.path.join(args.out_file,'result.csv'))
        final_results.append([config, df, alpha_beams_out])
        final_results_1.append([config, df, []])
        print('Done.')
        pickle_data(os.path.join(args.out_file,'result_1.pkl'), final_results_1)
        # pickle_data(os.path.join(args.out_file,'result.pkl'), final_results)
        print('Done.')

    