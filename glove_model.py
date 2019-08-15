# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:16:44 2019
@author: Sandy HE
"""

#importing the glove library
from glove import Corpus, Glove
import pandas as pd
from tqdm import tqdm
pruned_tagset = pd.read_csv("termstr_all.csv",index_col=0)
pruned_tagset = pruned_tagset[pruned_tagset['termstr'].notnull()]
tqdm.pandas(desc="split tagset string")
pruned_tagset = list(pruned_tagset['termstr'].progress_apply(lambda x: x.split(';')))
#creating a corpus object
corpus = Corpus() 

#training the corpus to generate the co occurence matrix which is used in GloVe
#
corpus.fit(pruned_tagset, window=3)
corpus.save('corpus.model')

#creating a Glove object which will use the matrix created in the above lines to create embeddings
#We can set the learning rate as it uses Gradient Descent and number of components
glove = Glove(no_components=150, learning_rate=0.05)
 
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
glove.save('glove.model')

#print(glove.dictionary)
termvec = glove.word_vectors
termdic = glove.dictionary
temp1 = glove.most_similar('rock', number=10)
print(temp1)
import pickle
with open('glovevec.pickle','wb') as file:
    pickle.dump(termvec,file)
with open('glovedict.pickle','wb') as file:
   pickle.dump(termdic,file)