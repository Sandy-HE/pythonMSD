"""
This is for tags preprocessing.
The following steps could be applied to normalize text as requirement.
>removing html tags
>removing accented characters
>expanding contractions 
>removing special characters
>removing stopword
>lemmatization or stemming

After processing, two files are output.
>termlist<index_interval>.csv: Dataframe<tags,popularity> for each song
>freqstat<index_interval>.csv: Dataframe<term,popularity> for each term
"""
import pandas as pd
import numpy as np

#Removing html tags
from bs4 import BeautifulSoup
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text
#test for removing html tags
#print(strip_html_tags('<html><h2>Strip HTML tags</h2></html>'))
 
#Removing accented characters  
import unicodedata
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text
#test for removing accented characters  
#print(remove_accented_chars('Sómě Áccěntěd těxt') )

#Expanding contractions
from utility import *
import re
def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

#test for expanding contractions
#print(expand_contractions("Y'all can't expand contractions I'd think"))

#Removing special characters
def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, ' ', text)
    return text

#test for removing special characters
#print(remove_special_characters("-chill-trip-lounge-down-; 00's;20th century ;-0220-", 
#                          remove_digits=True))



import nltk
#only run once and download all corpus
#nltk.download()

#Removing stopwords but exclude negation words
from nltk.corpus import stopwords 
stop_words=stopwords.words('english')
stop_words.remove('no')
stop_words.remove('not')

from nltk.tokenize.regexp import RegexpTokenizer
from nltk.tokenize import word_tokenize
#tokenizer = RegexpTokenizer(";",gaps=True)
def remove_stopwords(text):
    #term_tokens = tokenizer.tokenize(text)
    str = ' '
    text = remove_special_characters(text)
    word_tokens = np.char.split(text).tolist()
    filtered_words = []
    for wtoken in word_tokens:
        wtoken = wtoken.strip()
        if len(wtoken) >1 and wtoken not in stop_words:
            #wtoken = lemmatizer.lemmatize(wtoken,pos='v')
            filtered_words.append(wtoken)
    filtered_text = str.join(filtered_words)    
    return filtered_text

#test for removing stop words and meaningless words
#print(remove_stopwords("I am not 00's"))

#terms lemmatization
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer() 
def lemmatize_text(text):
    str=' '          
    word_tokens = np.char.split(text).tolist()
    lemma_words = []
    for wtoken in word_tokens:
        #wtoken = wtoken.strip().lower()            
        wtoken = lemmatizer.lemmatize(wtoken,pos='v')
        lemma_words.append(wtoken)
    lemma_terms = str.join(lemma_words)
    return lemma_terms

#test for lemmatizing terms
#print(lemmatize_text("increases crashing; his crashed yesterday; ours crashes daily"))


#Stemming terms
from nltk.stem import PorterStemmer
stemmer = PorterStemmer() 
def stemming_text(text):
    str=' '
    word_tokens = np.char.split(text).tolist()
    stem_words = []
    for wtoken in word_tokens:
        #wtoken = wtoken.strip().lower()            
        wtoken = stemmer.stem(wtoken)
        stem_words.append(wtoken)
    stem_terms = str.join(stem_words)
    return stem_terms    

#test for stemming terms
#print(stemming_text("increases crashing; his crashed yesterday; ours crashes daily"))
 
#integrate all tags processing function into this function
#input is a string
#output is a normalized string  
def normalize_corpus(term, html_stripping=False, contraction_expansion=True,
                     accented_char_removal=True, tags_lower_case=True, 
                     tags_lemmatization=True, special_char_removal=True, 
                     stopword_removal=True, remove_digits=False):
    #print("start normalizing corpus")
    #normalized_corpus = []
    # normalize tags in the corpus for each song track
    #for songtags in tags_dataset:
        # strip HTML
    if html_stripping:
        term = strip_html_tags(term)
    # remove accented characters
    if accented_char_removal:
        term = remove_accented_chars(term)
    # expand contractions    
    if contraction_expansion:
        term = expand_contractions(term)
    # lowercase the text    
    if tags_lower_case:
        term = term.lower()            
    # remove extra newlines
    #songtags = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
    
    # remove stopwords
    if stopword_removal:
        term = remove_stopwords(term)
    # lemmatize text
    if tags_lemmatization:
        term = lemmatize_text(term)    
    #normalized_corpus.append(term)
    
    return term   

#main entry to tag processing
def tags_handler(tagsets):
    termfreqlist = np.array(np.char.split(tagsets,sep=';').tolist())
    termfreqlist =np.char.rsplit(termfreqlist,sep='*',maxsplit=1).tolist()
    termfreqdf = pd.DataFrame(termfreqlist, columns=['term','popularity'])
    termfreqdf = termfreqdf[pd.to_numeric(termfreqdf['popularity'])!=0]
    termfreqdf['term'] = termfreqdf['term'].apply(normalize_corpus)
    termfreqdf = termfreqdf[termfreqdf['term']!='']
    return termfreqdf


tagsdata = pd.read_csv("alltranstags_phrase.csv")
import time
from tqdm import tqdm
start = time.perf_counter()
#tagsdata1 = tagsdata[0:105000]
print("Start to process tags...")
tqdm.pandas(desc="tag processing")
result = tagsdata['tags'].progress_apply(tags_handler)
tagsdata_result = result.to_frame()

#tagsdata_result = pd.concat([tagsdata_result,result.to_frame()])
end = time.perf_counter()
print('runtime:', round(end-start))

#transform the terms with frequency to a string
#input is a dataframe for one song, column is term and popularity
#str =';'
#def concat_terms(row):
#    return str.join([row['term']]*int(row['popularity']))
#
#def handle_result(item): 
#    termslist = item.apply(concat_terms,axis=1)
#    return str.join(termslist)
#
##accumulate all terms and frequency
df = pd.DataFrame(columns=['term','popularity'])
def term_freqstat(item):
    item['popularity'] = item['popularity'].apply(int)
    global df
    df = pd.concat([df,item])    

#output1: concat all tags of each song to one string
#term2str = tagsdata_result1['tags'].apply(handle_result)
#term2str = term2str.to_frame()
#term2str.columns = ['result']
#term2str.to_csv("tagspart2.csv")

#output2: get the terms set for each song without popularity
#in this way, we can save this result and restore it easily 
def get_terms(item):
    return list(item['term'].values)

def get_popularity(item):
    return list(item['popularity'].values)

termlist = tagsdata_result['tags'].apply(get_terms).to_frame()
termlist['popularity'] = tagsdata_result['tags'].apply(get_popularity)
termlist.to_csv("termlist_0_504555.csv")

#output3: calculate overall terms and frequency
#from tqdm import tqdm
tqdm.pandas(desc="termfreq acc")
tagsdata_result['tags'].progress_apply(term_freqstat)
grouped = df['popularity'].groupby(df['term'])
newtftable = grouped.sum()
df = pd.DataFrame({'term':newtftable.index,'popularity':newtftable.values})
df.to_csv("freqstat_0_504555.csv")

