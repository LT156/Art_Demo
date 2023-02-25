import numpy as np
import pandas as pd
import torch
from artemis.evaluation_fixed.single_caption_per_image import apply_reward

@torch.no_grad()
def get_reward(unique_id,emo,res,captions,**kwargs):
    gt_info_map = kwargs.get('gt_info_map', {})  
    word_json = kwargs.get('word_json', {})
    beam_size = kwargs.get('beam_size', 5)
    eval_metrics=kwargs.get('eval_metrics', ['ciderD'])

    caption_beams_out, _,_=res
    device=caption_beams_out.device

    batch_size,L=caption_beams_out.shape[0],caption_beams_out.shape[2]

    gt,gen,emo,u_ids=get_generated_and_real(unique_id,emo,caption_beams_out,**kwargs)
    captions=[  ' '.join(str(w) for w in captions[id].tolist() if w not in [0,1,2]) for id in range(len(captions)) for i in range(beam_size)]

    reward,reward_baseline,lcs_ids= compute_metrics(gt,gen,captions,emo,device,**kwargs)
    
    fixed_reward=np.zeros((batch_size*beam_size,L))
    #遍历每一个生成的句子，情感反义词-1，情感对应词及关键词+1
    if 'keywords_fix' in eval_metrics:
        keywords_reward=np.zeros((batch_size*beam_size,L))
        for i,u_id in enumerate(u_ids):
            if i%5==0:
                key_w=gt_info_map[u_id]['key_idx']
                emo=gt_info_map[u_id]['emotion']
                emo_w=word_json['emo_ids'][str(emo)]
                if emo <4:
                    emotion_antonym=word_json['polar_ids']['negative_words']
                elif emo <8:
                    emotion_antonym=word_json['polar_ids']['positive_words']
                else:
                    emotion_antonym=[]
            s_reward_=[]
            s_punishment_=[]
            s_=list(map(int, gen[i].split()))
            for j,w in enumerate(s_):
                if w in key_w+emo_w:
                    s_reward_.append(j)
                elif w in emotion_antonym:
                    s_punishment_.append(j)
            n=len(s_reward_)
            m=len(s_punishment_)
            if n+m!=0:
                if m==0:#只有奖励
                    m=len(s_)-n
                    appr=1-(n-m)/(n+m)
                    inac=-1-(n-m)/(n+m)
                    for idx in range(len(s_)):
                        if idx in s_reward_:
                            keywords_reward[i,idx]=appr
                        else:
                            keywords_reward[i,idx]=inac
                elif n==0:#只有惩罚
                    n=len(s_)-m
                    appr=1-(n-m)/(n+m)
                    inac=-1-(n-m)/(n+m)
                    for idx in range(len(s_)):
                        if idx in s_punishment_:
                            keywords_reward[i,idx]=inac
                        else:
                            keywords_reward[i,idx]=appr
                            
                else:#有奖有惩
                    appr=1-(n-m)/(n+m)
                    inac=-1-(n-m)/(n+m)  
                    for idx in range(len(s_)):
                        if idx in s_reward_:
                            keywords_reward[i,idx]=appr
                        elif idx in s_punishment_:
                            keywords_reward[i,idx]=inac
        fixed_reward=fixed_reward+0.01*keywords_reward

    
    if lcs_ids!=[]:
        lcs_reward=np.zeros((batch_size*beam_size,L))
        for i in range(batch_size*beam_size):
            if len(lcs_ids[i])!=0:
                c_len=len(list(map(int, captions[i].split())))
                #不惩罚第一个字
                if 0 not in lcs_ids[i]:
                    lcs_ids[i].append(0)

                n=len(lcs_ids[i])
                m=c_len-n
                appr=1-(n-m)/(n+m)
                inac=-1-(n-m)/(n+m)            
                for idx in range(c_len):
                    if idx in lcs_ids[i]:
                        lcs_reward[i,idx]=appr
                    else:
                        lcs_reward[i,idx]=inac
        fixed_reward=fixed_reward+0.01*lcs_reward
    fixed_reward=fixed_reward.reshape(batch_size,beam_size,L)
    return reward,reward_baseline,fixed_reward


@torch.no_grad()
def compute_metrics(gt,gen,captions,emo,device,**kwargs):
    gt_caption=pd.Series(gt)
    gen_caption=pd.Series(gen)
    if emo is not None:
        ref_emotions=pd.Series(emo)
    else:
        ref_emotions=None
    captions_=pd.Series(captions)

    cached_tokens=kwargs.get('cached_tokens', None)
    beam_size=kwargs.get('beam_size', 5)
    eval_metrics=kwargs.get('eval_metrics', ['ciderD'])
    txt2emo_clf=kwargs.get('txt2emo_clf', None)
    text2emo_vocab=kwargs.get('text2emo_vocab', None)
    vocab=kwargs.get('vocab', None)
    if ref_emotions is None:
        eval_metrics=[m for m in eval_metrics if m!="emo_alignment"]
    gen_scores= apply_reward(gen_caption, gt_caption, ref_emotions, captions_,cached_tokens,txt2emo_clf, vocab, text2emo_vocab,
                 device, methods_to_do=eval_metrics)

    #reward
    reward=np.zeros((len(gt),))
    reward_baseline=np.zeros((len(gt)//beam_size,))
    avg_weight=1/len([m for m in eval_metrics if m!='lcs'])
    if 'cider' in eval_metrics:
        cider_reward=np.array(pd.Series(gen_scores['cider']))
        reward_baseline=reward_baseline+avg_weight*np.mean(cider_reward.reshape((-1,beam_size)),axis=1)
        reward=reward+avg_weight*cider_reward
    if 'ciderD' in eval_metrics:
        ciderD_reward=np.array(pd.Series(gen_scores['ciderD']))
        reward_baseline=reward_baseline+2*np.mean(ciderD_reward.reshape((-1,beam_size)),axis=1)
        reward=reward+2*ciderD_reward
    if 'spice' in eval_metrics:
        spice_reward=np.array(pd.Series(gen_scores['spice']))
        reward=reward+avg_weight*spice_reward
    if 'meteor' in eval_metrics:
        meteor_reward=np.array(pd.Series(gen_scores['meteor']))
        reward=reward+avg_weight*meteor_reward
    if 'rouge' in eval_metrics:
        rouge_reward=np.array(pd.Series(gen_scores['rouge']))
        reward=reward+avg_weight*rouge_reward
    if 'bleu' in eval_metrics:
        bleu_reward=np.array(pd.Series(gen_scores['BLEU-3']))
        reward=reward+avg_weight*bleu_reward
    if 'emo_alignment' in eval_metrics:
        emo_reward=np.array(pd.Series(gen_scores['emo_alignment']))
        reward_baseline=reward_baseline+5*np.mean(emo_reward.reshape((-1,beam_size)),axis=1)
        reward=reward+5*emo_reward  

    lcs_ids=[]
    if 'lcs' in eval_metrics:
        lcs_ids=gen_scores['lcs']
    
    reward=reward.reshape((-1,beam_size))
    reward_baseline=reward_baseline[:,None]
    return reward,reward_baseline,lcs_ids


def get_generated_and_real(unique_id,emo,caption_beams_out,**kwargs):
    gt_info_map = kwargs.get('gt_info_map', {})  
    beam_size = kwargs.get('beam_size', 5)

    u_ids=[u_id  for u_id in unique_id for i in range(beam_size)]
    if emo is not None:
        emo=[idx  for e in emo for idx,v in enumerate(e) for i in range(beam_size) if v==1]
    else:
        emo=[]
    gt=[gt_info_map[idx]['idx_str']  for idx in unique_id for i in range(beam_size)]
    gen=[' '.join([str(int(idx))  for idx in scent[i]  if int(idx) not in [0,1,2]])  for scent in caption_beams_out for i in range(beam_size)]
    
    return gt,gen,emo,u_ids
    
