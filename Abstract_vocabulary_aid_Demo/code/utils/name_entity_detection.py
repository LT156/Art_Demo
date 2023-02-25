import nltk
from  nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

basic_punct = '()&`\'\"’.?!,:;/\-~*_=[–]{}$^@|%#<—>'
punct_to_space = str.maketrans(basic_punct, ' ' * len(basic_punct))  # map punctuation to space
stop_words = set(stopwords.words("english"))
stop_words2=['scævola','<unk>','unk',
            'become','becomes','becoming','became','get','got','gotten','getting','gets',
            'look','looks','looking','looked','seem','seems','seeming','seemed',
            'turn','turns','turning','turned','sound','sounds','sounding','sounded','sounder','soundest',
            'smell','smells','smelling','smelled','smelt','smelled','feel','feels','feeling','felt',
            'keep','keeps','keeping','kept', 'must','musts','can','could','may','might',
            'ought','will','would','shall','should','need','needs','needing','needed',
            'dare','dares','daring','dared','go','goes','went','gone','going',
            'have','has','had','having','do','does','doing','did','done',
            'man','men','woman','women','people','person','guy','guys','artist','artists',
            'everyone','everything','someone','something','anyone','anything'
            'make','makes','making','made','give','gives','giving','given','see','sees','seeing','saw','seen',
            'painting','paintings','scene','scenes','picture','pictures','figure','figures','portrait','portraits',
            'image','images','imagery','imageries','sight','sights','view','views','photo','photos',
            'many','much','little','small','big','piece','pieces',
            'twenty','thirty','forty' ,'fifty' ,'sixty' ,'seventy',' eighty','ninety',
            'above','following','left','right','middle','outer','inner','front','upper','towards']
stop_words=set.union(stop_words, stop_words2)

entity_names_list=['PERSON','ORGANIZATION','LOCATION','DATE','TIME','MONEY','PERCENT','FACILITY','GPE']

def remove_symbol(text):
    clean_text = text.translate(punct_to_space)
    return clean_text

def tag_Pos(text):
    text=remove_symbol(text)
    tagged = nltk.pos_tag(word_tokenize(text))  #词性标注
    entities = nltk.chunk.ne_chunk(tagged)  #命名实体识别
    
    word_Pos=[]
    word_Entity=[]
    
    for tagged_tree in entities:
        # extract only chunks having NE labels
        if hasattr(tagged_tree, 'label'):
            entity_name = ' '.join(c[0] for c in tagged_tree.leaves()) #get NE name 
            entity_type = ' '.join(c[1] for c in tagged_tree.leaves())
            entity_name=tagged_tree.label() # get NE category
        else:
            entity_name,entity_type = tagged_tree
        word_Pos.append(entity_type)
        word_Entity.append(entity_name)
    
    important_Words=[]
    for idx,data in enumerate(word_Entity):
        word_filter=['JJ','JJR','JJS','NN','NNS','VB','VBD','VBG','VBN','VBP','VBZ']
        if word_Pos[idx] in word_filter and data not in stop_words and data not in entity_names_list:
            important_Words.append(data)
    return important_Words




if __name__=='__main__':
    str1='i love this face it reminds me of a face from’ 2019 but yet it is ancient'
    print(tag_Pos(str1))
   

