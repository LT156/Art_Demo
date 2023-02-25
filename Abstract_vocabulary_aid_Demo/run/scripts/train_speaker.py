import os
import json
import torch
import time
import numpy as np
import os.path as osp
from torch import nn
from termcolor import colored
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from artemis.utils.vocabulary import Vocabulary

from Abstract_vocabulary_aid_Demo.code.utils.opts import parse_train_speaker_arguments
from Abstract_vocabulary_aid_Demo.code.artemis_model.neural_models.word_embeddings import init_token_bias
from Abstract_vocabulary_aid_Demo.code.artemis_model.in_out.neural_net_oriented import save_state_dicts, load_state_dicts, torch_load_model
from Abstract_vocabulary_aid_Demo.code.artemis_model.in_out.basics import create_dir, create_logger
from Abstract_vocabulary_aid_Demo.code.artemis_model.in_out.neural_net_oriented import df_to_pytorch_dataset, read_preprocessed_data_df, seed_torch_code

from Abstract_vocabulary_aid_Demo.code.reinforcement_learning.beam_decoder_train_val import beam_search_train, beam_search_val
from Abstract_vocabulary_aid_Demo.code.model.xe_train_and_val import negative_log_likelihood, single_epoch_train
from Abstract_vocabulary_aid_Demo.code.model.show_attend_and_tell import describe_model

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2300000000


if __name__ == '__main__':
    '''
    args里要有特征的路径
    word特征并入df里
    修改dataloader
    模型里加入abstract_projection_net(1000,512)
    '''
    
    #1.数据准备
    args = parse_train_speaker_arguments(save_args=True)
    if args.random_seed != -1:
        seed_torch_code(args.random_seed)
    vocab = Vocabulary.load(osp.join(args.data_dir, 'vocabulary.pkl'))
    vocab_14996 = Vocabulary.load(osp.join(args.data_dir2, 'vocabulary.pkl'))
    print('Using a vocabulary of size', len(vocab))

    df = read_preprocessed_data_df(args, vocab)
    import numpy as np
    loaded = np.load('F:/work/style_emotion/words/source/new_output/words_feature2.npz',allow_pickle=True)
    df2= pd.DataFrame(loaded['feature_df'],columns=loaded['columns'])
    df2=df2[['painting','abstract_features']]
    df = pd.merge(df,df2,on='painting',how='inner')
 
    ##1.1 强化学习数据准备GT
    with open(args.gt_json,'r+') as file:
        gt_json=file.read()
    gt_info_map=json.loads(gt_json)

    #1.2 强化学习数据准备Special words
    with open(args.sprcial_word,'r+') as file:
        content=file.read()
    word_json=json.loads(content)#将json格式文件转化为python的字典文件
    
    # 1.3数据加载器
    data_loaders, _ = df_to_pytorch_dataset(df, args)
    print('Will use {} annotations for training.'.format(len(data_loaders['train'].dataset)))


    #2. 模型准备
    torch.backends.cudnn.benchmark = True
    #  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:',device)
    ## 2.1 文本分类器——文本分类reward
    txt2emo_clf = torch_load_model(args.text2emo_path, map_location=device)

    ## 2.2 模型定义
    model = describe_model(vocab, args)

    token_bias = init_token_bias(data_loaders['train'].dataset.tokens, vocab)
    model.decoder.next_word.bias = token_bias
    # 参数加载
    if args.resume_path:
        print('FT in the most aggressive way. Just let the speaker continue training...')
        loaded_epoch = load_state_dicts(args.resume_path, map_location='cpu', model=model)
        print('Loaded a pre-trained model at epoch {}.'.format(loaded_epoch))
    model.to(device)
    
    ## 损失函数与优化 Prepare Loss/Optimization
        ## Prepare Loss/Optimization
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam([
        {'params': filter(lambda p: p.requires_grad, model.encoder.parameters()), 'lr': args.encoder_lr},
        {'params': filter(lambda p: p.requires_grad, model.decoder.parameters()), 'lr': args.decoder_lr}])
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              factor=0.5,
                                                              patience=args.lr_patience,
                                                              verbose=True,
                                                              min_lr=5e-7)

    #参数定义： Misc.
    best_epoch = -1
    best_val_nll = np.Inf
    best_val_reward  = -np.Inf
    print_freq = 10
    start_training_epoch = 0
    no_improvement = 0
    tb_writer = SummaryWriter(create_dir(osp.join(args.log_dir, 'tb_log')))
    model_dir = create_dir(osp.join(args.log_dir, 'checkpoints'))
    logger = create_logger(args.log_dir)
    
    
    train_args = dict()
    train_args['use_emotion'] = args.use_emo_grounding
    train_args['use_abstract'] = args.use_abstract_data
    train_args['alpha_c'] = args.atn_cover_img_alpha

    rl_args = dict()
    rl_args['beam_size'] = args.beam_size
    rl_args['gt_info_map'] = gt_info_map
    rl_args['word_json'] = word_json
    rl_args['cached_tokens'] = args.cached_tokens
    rl_args['eval_metrics'] = args.eval_metrics
    rl_args['text2emo_vocab'] = vocab_14996
    rl_args['txt2emo_clf'] = txt2emo_clf

    
    #4.添加强化学习：
    scst_flag=args.use_scst_training
    load_model=False
    best_val_reward=0
    best_model=None
    if load_model:
        model_ckp=args.model_ckp
        epoch=load_state_dicts(model_ckp, model=model, map_location=device)
        with torch.no_grad():
            _,_,reward_baseline = beam_search_val(args.mode, model,data_loaders['val'],device,**rl_args)
            nll_loss=negative_log_likelihood(model, data_loaders['val'], device,args)
            best_val_reward=reward_baseline
            logger.warning('loadding model from {}'.format(model_ckp))
            logger.warning('loadding epoch-{} model for scst training'.format(epoch, args.train_patience))
            logger.warning('loaded model reward={}'.format(best_val_reward))
            logger.warning('loaded model nll_loss={}'.format(nll_loss))
    
    
    
    
    ## 训练
    logger.info('Starting the training of the speaker.')
    for epoch in range(start_training_epoch, args.max_train_epochs + 1):
        start_time = time.time()
        
        if scst_flag==False:
            # 常规训练-训练
            
            xe_loss = single_epoch_train(data_loaders['train'], model, criterion, optimizer, epoch, device,\
                                        print_freq=print_freq, tb_writer=tb_writer, **train_args)
            logger.info('Epoch {} entropy_loss {:.30f} time {:.1f}'.format(epoch, xe_loss, (time.time() - start_time) / 60))
            
            
            # 常规训练-验证
            val_nll = negative_log_likelihood(model, data_loaders['val'], device,args)
            
            # 常规训练-是否提升
            lr_scheduler.step(val_nll)
            tb_writer.add_scalar('training-entropy-per-epoch', xe_loss, epoch)
            tb_writer.add_scalar('val-nnl-per-epoch', val_nll, epoch)
            logger.info('train loss {xe_loss:.30f}'.format(xe_loss=xe_loss))
            if  val_nll < best_val_nll:
                logger.info('Validation loss {}, *improved* @epoch {}'.format(val_nll, epoch))
                best_val_nll = val_nll
                best_epoch = epoch
                best_model = model
                out_name = osp.join(model_dir,  'best_model_xe.pt')
                save_state_dicts(out_name, epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
                no_improvement = 0
            else:
                logger.info('Validation loss did NOT improve @epoch {}'.format(epoch))
                no_improvement += 1

            
            
        else:
            # 强化训练-训练
            scst_loss,reward,scst_reward_baseline = beam_search_train(args.mode, epoch,model, data_loaders['train'],device, optimizer,tb_writer=tb_writer,**rl_args)
            logger.info('Epoch reward_baseline {:.30f} time {:.1f}'.format(reward_baseline, (time.time() - start_time) / 60))
            
            # 强化训练-验证
            _,_,val_reward_baseline = beam_search_val(args.mode, model,data_loaders['val'],device,**rl_args)
            logger.info('Validation reward {val_reward:.30f}\t'.format(val_reward=val_reward_baseline))
            
            # 强化训练-是否提升
            lr_scheduler.step(val_reward_baseline)
            tb_writer.add_scalar('training-scst-reward-per-epoch', scst_reward_baseline, epoch)
            tb_writer.add_scalar('val-reward-per-epoch', val_reward_baseline, epoch)
            logger.info('train loss {scst_loss:.30f}\t train reward {train_reward:.30f}\t'.format(scst_loss=scst_loss,train_reward=scst_reward_baseline))
            if  val_reward_baseline > best_val_reward:
                logger.info('Validation reward {}, *improved* @epoch {}'.format(val_reward_baseline, epoch))
                best_val_reward= val_reward_baseline
                best_epoch = epoch
                best_model = model
                out_name = osp.join(model_dir,  'best_model_scst.pt')
                save_state_dicts(out_name, epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
                no_improvement = 0
            else:
                logger.info('Validation reward did NOT improve @epoch {}'.format(epoch))
                no_improvement += 1
            
            
        
            
           
            
        
        
        tb_writer.add_scalar('encoder-learning-rate-per-epoch', optimizer.param_groups[0]['lr'], epoch)
        tb_writer.add_scalar('decoder-learning-rate-per-epoch', optimizer.param_groups[1]['lr'], epoch)
        
        if args.save_each_epoch:
            out_name = osp.join(model_dir, 'model_epoch_' + str(epoch) + '.pt')
            save_state_dicts(out_name, epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
        
        #xe接scst
        if no_improvement == args.train_patience and not scst_flag:
            logger.warning('Stopping the training @epoch-{} due to lack of progress in '
                           'validation-reduction (patience hit {} '
                           'epochs'.format(epoch, args.train_patience))
            # 暂不进行强化训练
            break
            for name, parameter in model.encoder.resnet.named_parameters():
                parameter.requires_grad = False

            scst_flag=True
            no_improvement=0
            optimizer = torch.optim.Adam([
                {'params': filter(lambda p: p.requires_grad, model.encoder.parameters()), 'lr': args.encoder_lr},
                {'params': filter(lambda p: p.requires_grad, model.decoder.parameters()), 'lr': args.decoder_lr}])

            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                    factor=0.5,
                                                                    patience=args.lr_patience,
                                                                    verbose=True,
                                                                    min_lr=5e-7)
            #epoch=load_state_dicts(out_name, model=model, map_location=device)
            model=best_model
            
            logger.warning('loadding epoch-{} model for scst training'.format(best_epoch, args.train_patience))
            _,_,reward_baseline = beam_search_val(args.mode, model,val_data_loader, device,**rl_args)
            best_val_reward=reward_baseline
            logger.warning('loaded epoch-{} reward={}'.format(epoch, best_val_reward))

        if  no_improvement == args.train_patience and scst_flag:
            break
        
    with open(osp.join(model_dir, 'final_result.txt'), 'w') as f_out:
        msg = ('Best Validation NLL: {:.20f}\t Best Validation reward: {:.20f} (achieved @epoch {})'.format(best_val_nll,best_val_reward, best_epoch))
        f_out.write(msg)

    logger.info('Finished training properly.')
    tb_writer.close()
    