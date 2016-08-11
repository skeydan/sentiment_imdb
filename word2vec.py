from __future__ import division
import logging

import pandas as pd 
import numpy as np

import nltk.data
from gensim.models import word2vec

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

import preprocess
import utils


reviews = pd.read_csv("data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3, encoding='utf-8')

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# word2vec expects whole sentences
def review_to_sentences(review, tokenizer, remove_stopwords=False):
    review_sentences = tokenizer.tokenize(review.strip())
    review_sentences_clean = []
    for sentence in review_sentences:
        if len(sentence) > 0:
            review_sentences_clean.append(preprocess.cleanup(sentence))
    return review_sentences_clean
    
sentences = []  

i=0
for review in reviews["review"]:
    sentences += review_to_sentences(review, tokenizer)
    i = i+1
    if i % 1000 == 0: print 'Sentences cleaned: {}'.format(i)  
    
print len(sentences)
print sentences[0:5]


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count,
                          window = context, sample = downsampling)

model.init_sims(replace=True)

model_name = "300features_40minwords_10context"
model.save(model_name)

print model.vocab

model.doesnt_match("man woman child kitchen".split())
model.doesnt_match("france england germany berlin".split())
model.most_similar("awful")

print model.syn0.shape
print model['movie']

sections = model.accuracy('questions-words.txt')
correct, incorrect = len(sections[-1]['correct']), len(sections[-1]['incorrect'])
print('%s: %0.2f%% correct (%d of %d)' % (model, float(correct*100)/(correct+incorrect), correct, correct+incorrect))

model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
model.doesnt_match("breakfast cereal dinner lunch";.split())
model.similarity('woman', 'man')


'''
Now, how to combine the vectors for one review?

https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-3-more-fun-with-word-vectors tries 2 things:

- averaging over the features per review
- clustering the words and assigning review to cluster with highest word count 

Others:
- average of word vectors multiplied with tdf_idf

Alternatives:
- doc2vec
- sentence2vec
- paragraph vectors

Try: PCA!!!!

'''

