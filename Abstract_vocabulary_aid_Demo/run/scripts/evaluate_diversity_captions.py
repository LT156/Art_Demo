import json
import nltk
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pylab as plt
from collections import Counter
from functools import partial
from ast import literal_eval
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from artemis.utils.basic import df_parallel_column_apply
from artemis.analysis.utils import contains_word, contains_bigrams, concreteness_of_sentence, pos_analysis
from artemis.language.basics import tokenize_and_spell, ngrams
from artemis.language.part_of_speech import nltk_parallel_tagging_of_tokens
from artemis.utils.visualization import plot_overlayed_two_histograms
from artemis.utils.other_datasets.flickr30K_entities import load_all_linguistic_annotations
from artemis.utils.other_datasets.conceptual_captions import load_conceptual_captions
from artemis.utils.other_datasets.google_refexp import load_google_refexp_captions

try:
    from textblob import TextBlob    
except:
    print('For analyzing the subjectivity (bottom parts of notebook) you need to install textblob')  
    print('e.g., conda install -c conda-forge textblob')

freq_file = '/media/data/LuoTing/local/work_space/artemis-master/artemis/data/symspell_frequency_dictionary_en_82_765.txt'
glove_file = '/media/data/LuoTing/local/work_space/artemis-master/artemis/data/glove.6B.100d.vocabulary.txt'
brm_file=\
'/media/data/LuoTing/local/work_space/artemis-master/RAIVS/scripts/test/Concreteness_ratings_Brysbaert_et_al_BRM.xlsx'

##
## Hyper-params for notebook 
##
load_ela = True
brm_drop_articles = True # use or not the: "the", "a", "an", "every".
spell_check_non_artemis_data = False

#抽象字典准备：
brm_data = pd.read_excel(brm_file)
brm_data.Word = brm_data.Word.apply(lambda x: str(x).lower())
if brm_drop_articles:
    brm_data = brm_data[brm_data.Dom_Pos != 'Article'] 
brm_data = brm_data[['Word', 'Conc.M']]
brm_data.columns = ['word', 'concreteness']
word_to_concreteness = dict(zip(brm_data.word, brm_data.concreteness))

#抽象元组
# restrict to the most abstract words, bigrams (per percentile value). How many of those are precent in eaach dataset?
wc_vals = list(word_to_concreteness.values())
prc = 1
conc_threshold = np.percentile(wc_vals, prc)
abstract_unigrams = set()
abstract_bigrams = set()
for k, v in word_to_concreteness.items():
    if v <= conc_threshold:
        if ' ' in k:
            abstract_bigrams.add(k)
        else:
            abstract_unigrams.add(k)            
print('Percentile of most abstract:', prc)
print('Number of words/bigrams to be used as abstract:（抽象词，抽象元组）', len(abstract_unigrams), len(abstract_bigrams))

#抽象性分析：
def conreteness_Analysis(df):
    concreteness_score = df.tokens.apply(lambda x: concreteness_of_sentence(x, word_to_concreteness))    
    smean = concreteness_score.mean().round(2)
    smedian = concreteness_score.median().round(2)#能够返回给定数值的中值，
    print('concreteness(均值，中位数)', smean, smedian)

    uses_abstract = contains_word(df.tokens, abstract_unigrams)
    uses_abstract |= df.tokens.apply(partial(contains_bigrams, abstract_bigrams))
    print('使用抽象元组的均值：',uses_abstract.mean())
    return smean, smedian

#情感分析
def vader_classify(score, threshold=0.05):
    if abs(score) < threshold:
        return 'neutral'
    if score > 0:
        return 'positive'
    if score < 0:
        return 'negative'
    
def vader_score(sentence):
    return vader.polarity_scores(sentence)['compound']

def sentiment_score(df):
        
    scores = df['utterance_spelled'].apply(lambda x: vader.polarity_scores(x)['compound'])
    smean = scores.abs().mean().round(2)
    smedian = scores.abs().median().round(2)    
    print('sentiment-score（均值，中位数）', smean, smedian)

    sentiment_classe_list = []  # aggregate for all datasets to make a nice plot
    sentiment_classe = dict()
    threshold = 0.05

    temp = scores.apply(lambda x: vader_classify(x, threshold=threshold))
    sentiment_classe = temp    
    sentiment_classe_list.extend(temp.to_list())  

    #中性情感词均分
    print('中性情感词均分',(sentiment_classe == 'neutral').mean())

    return smean, smedian

#主观性分析
def subjectivity(utterance):
    testimonial = TextBlob(utterance)
    return testimonial.sentiment.subjectivity

def subjectivity_score(df):
 
    subjectivity_scores = df_parallel_column_apply(df, subjectivity, 'utterance_spelled')
    su_mean=subjectivity_scores.mean().round(2)
    su_median=subjectivity_scores.median().round(2)
    return su_mean,su_median

 #词性的丰富性
def rich_words(df):
    df['pos'] = nltk_parallel_tagging_of_tokens(df.tokens)
    result=pos_analysis(df)
    return result
if __name__=='__main__':
    save_path='/media/data/LuoTing/local/work_space/out/step_4/all_test/_key_model/diversity_evaluation.json'
    artemis_preprocessed_csv = '/media/data/LuoTing/local/work_space/out/step_3/all_test/_key_model/result.csv'




    #,art_style,painting,grounding_emotion,caption
    result_scores={}
    df = pd.read_csv(artemis_preprocessed_csv)
    df['utterance']=df['caption']
    caption_missed_tokens = tokenize_and_spell(df, glove_file, freq_file, 
                                            nltk.word_tokenize, spell_check=spell_check_non_artemis_data)

    #平均长度
    print(k, 'N-sentences:', len(df), 'Average Length:', df.tokens_len.mean().round(1))
    result_scores['Average Length']=df.tokens_len.mean().round(1)

    ##抽象分析
    smean, smedian=conreteness_Analysis(df)
    result_scores['concreteness_score']=(smean, smedian)

    ## 情感分析
    sentiment_scores = dict()
    vader = SentimentIntensityAnalyzer()
    smean, smedian=sentiment_score(df)
    result_scores['sentiment_scores']=(smean, smedian)

    ## 主观性分析：
    su_mean,su_median = subjectivity_score(df)
    result_scores['subjectivity_score']=(su_mean,su_median)

    ## 词性丰富性度量：
    result = rich_words(df)
    result_scores['rich_words']=result

    dict_json=json.dumps(result_scores)#转化为json格式文件
    #将json文件保存为.json格式文件
    with open(save_path,'w') as file:
        file.write(dict_json)



    
