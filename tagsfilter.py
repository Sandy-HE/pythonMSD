# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:41:09 2019

@author: 13035338
"""
import pandas as pd
import numpy as np

TERM_COUNT_MIN = 1000
DOC_PROP_MAX = 0.8
DOC_PROP_MIN = 0.0002

#a) get term frequency dataframe
#b) remove low-frequency terms based on the threshold 'TERM_COUNT_MIN'
#c) calculate doc freqency for each term
#d) based on doc freqency proportion, remove some high and low frequency terms
#e) based on final pruned terms, refine tags data


#Collect the preprocessed data 
#The data is saved in several files, we need to concatenate them together
#freqstat_1 = pd.read_csv("freqstat_0_105000.csv",index_col=0)
#freqstat_2 = pd.read_csv("freqstat_105000_210000.csv",index_col=0)
#freqstat_3 = pd.read_csv("freqstat_210000_315000.csv",index_col=0)
#freqstat_4 = pd.read_csv("freqstat_315000_420000.csv",index_col=0)
#freqstat_5 = pd.read_csv("freqstat_420000_504555.csv",index_col=0)
#freqstatall = pd.concat([freqstat_1,freqstat_2,freqstat_3,freqstat_4,freqstat_5])
#
#grouped = freqstatall['popularity'].groupby(freqstatall['term'])
#newtftable = grouped.sum()
#freqstatall = pd.DataFrame({'term':newtftable.index,'popularity':newtftable.values})
#freqstatall.to_csv("freqstat_all.csv")

import ast
def to_np_array(array_string):
    #array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))
#termsets1 = pd.read_csv("termlist_0_105000.csv",index_col=0)
#termsets2 = pd.read_csv("termlist_105000_210000.csv",index_col=0)
#termsets3 = pd.read_csv("termlist_210000_315000.csv",index_col=0)
#termsets4 = pd.read_csv("termlist_315000_420000.csv",index_col=0)
#termsets5 = pd.read_csv("termlist_420000_504555.csv",index_col=0)
#
#termsets = pd.concat([termsets1,termsets2,termsets3,termsets4,termsets5])
#termsets.to_csv("termlist_all.csv")

#a DataFrame of term frequency based on each term
freqstatall = pd.read_csv("freqstat_all.csv",index_col=0)
#a DataFrame of lists of tags and popularity based on each song
termsets = pd.read_csv("termlist_all.csv",index_col=0,converters={'tags': to_np_array,'popularity': to_np_array})

#input: corpus is a term list; 
#input: percentage means calculating frequency proportion or not
#output: termfreqs is a Series of term frequency
#def get_term_freq(corpus, percentage = False):
#    if percentage:
#        termfreqs = pd.Series(corpus).value_counts(normalize=True)
#        termfreqs.name='termfreq_prop'
#    else:
#        termfreqs = pd.Series(corpus).value_counts()
#        termfreqs.name='termfreq'
#    return termfreqs

#input: statdata is a DataFrame with term and popularity
#input: term_freq_min means the minimum of term frequency
#output: a dataframe of term frequency without low frequency
def remove_lowfreq_terms(statdata, term_freq_min = TERM_COUNT_MIN):
    return statdata[statdata['popularity'] >= term_freq_min]

#input: termlist is a numpy array of terms
#input: docterms is a DataFrame of terms for all songs
#output: docfreqs is a series of document frequency
from tqdm import tqdm
def get_doc_freq(termlist, docterms):
    docfreqs=[]
    for term in tqdm(termlist, desc='get doc freq:'):        
        docfreq = docterms[list(docterms['tags'].apply(lambda x: term in x))].shape[0]
        docfreqs.append(docfreq)
    return docfreqs

#input: stattable is a DataFrame containing terms statistic info
#input: doc_prop_max means maximum document proportion including this term
#input: doc_prop_min means minimum document proportion including this term
#output: a series of pruned terms(index kept)
def remove_extreme_docfreq_terms(stattable, doc_prop_max=DOC_PROP_MAX, doc_prop_min=DOC_PROP_MIN):
    stattable = stattable[(stattable.docfreq_prop <= doc_prop_max) & (stattable.docfreq_prop >= doc_prop_min) ]
    return stattable['term']

#main entry for filtering terms
#input: statdata is a dataframe of term popularity for each term
#input: tagsets is a dataframe of tag set for each song
#output: a series of pruned tag terms
def prune_terms(statdata, tagsets,term_freq_min = TERM_COUNT_MIN, doc_prop_max=DOC_PROP_MAX, doc_prop_min=DOC_PROP_MIN):
        
    print("start removing low term freq")
    termstat = remove_lowfreq_terms(statdata)
    
    termlist = termstat['term'].values
    
    print("start getting doc freq")
    docfreqs = get_doc_freq(termlist,tagsets)
    
    termstat['docfreq'] = docfreqs
    #termstat.to_csv("term_statistic.csv")
    
    songsum = len(tagsets)
    termstat['docfreq_prop'] = termstat['docfreq']/songsum
    
    print("start removing extreme doc freq terms")
    pruned_vocab = remove_extreme_docfreq_terms(termstat)
    return pruned_vocab

pruned_taglist = prune_terms(freqstatall,termsets)
#termstat = pd.read_csv("term_statistic.csv",index_col=0) 
#pruned_taglist = remove_extreme_docfreq_terms(termstat)
    
print("start shaping corpus")
def concat_terms(row):
    return ';'.join([row['tags']]*int(row['popularity']))

def filter_tags(item):
    termfreqdf = pd.DataFrame({'tags':item['tags'],'popularity': item['popularity']})
    termfreqdf = termfreqdf[termfreqdf['tags'].apply(lambda x: x in pruned_taglist.values)]
    termline = termfreqdf.apply(concat_terms,axis=1)
    return ';'.join(termline)

result = termsets.apply(filter_tags, axis=1)
result = result.to_frame(name='termstr')
result.to_csv("termstr_all.csv")
    