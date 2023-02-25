import numpy as np
import json
import os.path as osp
import pprint
import pandas as pd
import argparse
from artemis.utils.vocabulary import Vocabulary
from RAIVS.utils.name_entity_detection import tag_Pos
from collections import Counter
from artemis.in_out.basics import pickle_data
from RAIVS.artemis_model.in_out.neural_net_oriented import literal_eval


def group_gt_annotations(preprocessed_dataframe, vocab):
    df = preprocessed_dataframe
    results = dict()
    for split, g in df.groupby('split'): # group-by split
        g.reset_index(inplace=True, drop=True)
        #根据unique_id聚类：情感、tokens、words
        g = g.groupby(['unique_id']) 
        refs_pre_vocab_grouped = g['utterance_spelled'].apply(list).reset_index(name='references_pre_vocab')
        emotion_grouped = g['emotion_label'].apply(lambda x :list(x)[0]).reset_index(name='emotion')
        tokens_grouped = g['tokens_encoded'].apply(list).reset_index(name='tokens_encoded')
        references_grouped =g['tokens_encoded'].apply(lambda x: [vocab.decode_print(sent) for sent in list(x)]).reset_index(name='references')
        # join results in a new single dataframe
        tmp = pd.merge(refs_pre_vocab_grouped, emotion_grouped)
        tmp = pd.merge(tmp, references_grouped)
        result = pd.merge(tmp, tokens_grouped)
        results[split] = result
    return results

def parse_arguments(notebook_options=None):
    parser = argparse.ArgumentParser(description='Preprocess content data')
    #设置服务器训练时是否需要数据
    flag=False
    raw_data_csv=r'F:\work\Image_emotion_analysis\artemis-master\small_out\step1\artemis_preprocessed.csv'
    save_out_dir=r'F:\work\Image_emotion_analysis\artemis-master\small_out\step0'
    vocab_dir=r'F:\work\Image_emotion_analysis\artemis-master\small_out\step1'
    # Required arguments
    parser.add_argument('-save-out-dir', type=str, required=flag, default=save_out_dir,help='where to save the processed data')
    parser.add_argument('-vocab-dir', type=str, required=flag, default=vocab_dir,help='vocab dir')
    parser.add_argument('-raw-data-csv', type=str, required=flag, default=raw_data_csv,help='content csv')
    
    if notebook_options is not None:  # Pass options directly (useful inside say jupyter)
        args = parser.parse_args(notebook_options)
    else:
        args = parser.parse_args() # Read from command line.

    args_string = pprint.pformat(vars(args))
    print(args_string)
    return args

#获得关键词的排序
def get_keywords_sort(important_info):
    counter = Counter()
    for word_list in important_info:
        counter.update(word_list)
    tuples_sorted=sorted(counter.items(), key=lambda x:x[1],reverse=True)
    words_map=dict(tuples_sorted)
    return words_map
#获得情感词汇
def get_emo_words(df):
    emo_words={}
    emo_p_words={0:[],1:[],2:[]}#积极、消极、其他
    for emo,g in df.groupby('emotion_label'):
        g.reset_index(inplace=True, drop=True)
        #通过情感对应的词汇、sorted、添加词
        counter = Counter()
        for word_list in g['import_words']:
            counter.update(word_list)
        tuples_sorted=sorted(counter.items(), key=lambda x:x[1],reverse=True)
        #通过情感极性，添加词汇
        emo_words[emo]=[w for w,c in tuples_sorted] 
        if emo in [0,1,2,3]:
            emo_p_words[0]=emo_p_words[0]+emo_words[emo]
        elif emo in [4,5,6,7]:
            emo_p_words[1]=emo_p_words[1]+emo_words[emo]
        else:
            emo_p_words[2]=emo_p_words[2]+emo_words[emo]

    #各情感词汇去重，保留各种情感独立的描述词，每个情感最多保留100个
    emo_special_words={}
    for key,value in emo_words.items():
        other_list=[]
        for emo in emo_words.keys():
            if emo!=key:
                other_list=other_list+emo_words[emo]
        other_list=list(set(other_list))
        diff_list=[v for v in value if v not in other_list]
        emo_special_words[key]=diff_list[:100]

    #各情感极性词汇去重
    result_p1=[v for v in emo_special_words[0] if v not in emo_special_words[1]+emo_special_words[2]]
    result_p2=[v for v in emo_special_words[1] if v not in emo_special_words[0]+emo_special_words[2]]
    result_p3=[v for v in emo_special_words[2] if v not in emo_special_words[0]+emo_special_words[1]]
            
        
    #极性词汇，再将极性词汇根据词频重排，各取100词
    polar_words={}
    polar_words['other_words']=[v for v in result_p1 if v  in result_p2 and v in result_p3 ]
    polar_words['other_words']={v:words_map[v] for v in polar_words['other_words']}
    polar_words['other_words']=dict(sorted(polar_words['other_words'].items(), key=lambda item: item[1],reverse=True)[:100])

    polar_words['positive_words']=[v for v in result_p2 if v not in result_p1 and v not in result_p3]
    polar_words['positive_words']={v:words_map[v] for v in polar_words['positive_words']}
    polar_words['positive_words']=dict(sorted(polar_words['positive_words'].items(), key=lambda item: item[1],reverse=True)[:100])

    polar_words['negative_words']=[v for v in result_p3 if v not in result_p1 and v not in result_p2]
    polar_words['negative_words']={v:words_map[v] for v in polar_words['negative_words']}
    polar_words['negative_words']=dict(sorted(polar_words['negative_words'].items(), key=lambda item: item[1],reverse=True)[:100])
   
    
    special_words={}
    #情感词汇
    special_words['emo_words']=emo_special_words
    special_words['emo_ids']={emo:[vocab.word2idx[w] for w in value]for emo,value in emo_special_words.items() }
    #极性词汇
    special_words['polar_words']=polar_words
    special_words['polar_ids']={emo:[vocab.word2idx[w] for w in value]for emo,value in polar_words.items() }
    #常见词汇
    special_words['important_words']=list(words_map.keys())[:300]
    special_words['important_ids']=[vocab.word2idx[w] for w in special_words['important_words']]

    dict_json=json.dumps(special_words)#转化为json格式文件
    return  dict_json

def get_emo_group(df):
    #构造gt
    groups_split = group_gt_annotations(df, vocab)
    #构建img_emo_gt.dict
    reference_dict={}
    for split,df in groups_split.items():
        for idx,row in df.iterrows():
            items={}
            id=row['unique_id']
            items['idx_str']=[" ".join([str(id) for id in sent if id not in [0,1,2]]) for sent in row['tokens_encoded']]
            items['gt_str']=row['references']
            items['emotion']=row['emotion']
            '''
            #这里key_words和df['important_words']不是一样的嘛
            items['key_words']=list(set([word for sent in row['references'] for word in tag_Pos(sent) if word in vocab.word2idx.keys()] ) )
            items['key_idx']=[vocab.word2idx[word] for word in items['key_words']]
            '''
            reference_dict[id]=items
            

    json_data = json.dumps(reference_dict)
    return json_data,groups_split
    
    

if __name__=='__main__':
    args=parse_arguments()

    vocab = Vocabulary.load(osp.join(args.vocab_dir, 'vocabulary.pkl'))
    df=pd.read_csv(args.raw_data_csv)
    df.tokens=df.tokens.apply(literal_eval)
    df.tokens_encoded=df.tokens_encoded.apply(literal_eval)
    words=vocab.word2idx.keys()
    #添加额外额属性列：unique_id,import_words
    df['unique_id']=df['painting']
    df.to_csv(osp.join(args.save_out_dir,'artemis_preprocessed_enhance.csv'))
    print(f'Done. artemis_preprocessed_enhance saved')

    '''
    
    #关键词清洗
    important_info=df['utterance_spelled'].apply(lambda x:[v for v in tag_Pos(x) if v in words])
    
    #关键词词频统计map
    words_map=get_keywords_sort(important_info)
    df['import_words']=important_info
    #根据情感分组词汇，各情感中词频排序
    dict_json=get_emo_words(df)
    with open(osp.join(args.save_out_dir,'special_words.json'),'w') as file:
         file.write(dict_json)
    print(f'Done. special_words saved')
    '''
    

    json_data,groups_split=get_emo_group(df)
    with open(osp.join(args.save_out_dir, 'artemis_emogt_encoded_enhance.json'), 'w') as f_six:
        f_six.write(json_data)
    pickle_data(osp.join(args.save_out_dir, 'artemis_gt_references_enhance_grouped.pkl'), groups_split)
    print(f'Done. Check saved results in provided save-out-dir: {args.save_out_dir}')
        