from __future__ import division
import logging

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

import nltk.data
from gensim.models import word2vec
import io

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import preprocess
import utils

model_exists = True

with io.open('data/aclImdb/train-pos.txt', encoding='utf-8') as f:
    l = list(f)
    train_pos = pd.DataFrame({'review': l})
    
with io.open('data/aclImdb/train-neg.txt', encoding='utf-8') as f:
    l = list(f)
    train_neg = pd.DataFrame({'review': l})    
    
train_reviews = pd.concat([train_neg, train_pos], ignore_index=True)


with io.open('data/aclImdb/test-pos.txt', encoding='utf-8') as f:
    l = list(f)
    test_pos = pd.DataFrame({'review': l})
    
with io.open('data/aclImdb/test-neg.txt', encoding='utf-8') as f:
    l = list(f)
    test_neg = pd.DataFrame({'review': l})    
    
test_reviews = pd.concat([test_neg, test_pos], ignore_index=True)


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

sentences_train = []  

for review in train_reviews["review"]:
    sentences_train += [s.split() for s in tokenizer.tokenize(review)]

    
print len(sentences_train)
print sentences_train[0]

sentences_test = []  

for review in test_reviews["review"]:
    sentences_test += [s.split() for s in tokenizer.tokenize(review)]
    
print len(sentences_test)
print sentences_test[0]

all_sentences = sentences_train + sentences_test

if not model_exists:

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    num_features = 100    # Word vector dimensionality                      
    min_word_count = 20   # Minimum word count                        
    num_workers = 2       # Number of threads to run in parallel
    context = 10          # Context window size                                                                                    
    downsampling = 1e-3   # Downsample setting for frequent words

    model = word2vec.Word2Vec(all_sentences, workers=num_workers, size=num_features, min_count = min_word_count,
                              window = context, sample = downsampling)

    model.init_sims(replace=True)

    model_name = "models/word2vec_100features"
    model.save(model_name)

else:    
    model = word2vec.Word2Vec.load('models/word2vec_100features')
    
print {k: model.vocab[k] for k in model.vocab.keys()[:5]}
print model.syn0.shape
print model['movie']

model.similarity('awesome', 'awful')

for word in ['awful', 'awesome']:  
    print '\n\nSimilar words to: {}\n'.format(word)  
    similar = model.most_similar(word, topn=10)
    print 'Model: {}\n{}\n'.format(model, similar)

print 'Model: {}: {}\n'.format(model, model.most_similar(positive=['awesome'], negative=['awful']))

model.doesnt_match("good bad awful terrible".split())

sections = model.accuracy('data/questions-words.txt')
correct, incorrect = len(sections[-1]['correct']), len(sections[-1]['incorrect'])
print('%s: %0.2f%% correct (%d of %d)' % (model, float(correct*100)/(correct+incorrect), correct, correct+incorrect))


########################################################################

words = ['story', 'movie','plot', 'film', 'good', 'bad', 'awful', 'awesome', 'man', 'woman', 'like', 'actor', 'actress']
vectors = model.syn0

pca = PCA(n_components=2)
pca_2d = pca.fit_transform(vectors)

'''   
tsne = TSNE(n_components=2, random_state=0, verbose=10, init='pca')
tsne_2d = tsne.fit_transform(vectors)
'''

first = True

#for name, transform in zip(['PCA', 'TSNE'], [pca_2d, tsne_2d]):  
for name, transform in zip(['PCA'], [pca_2d]):   
    plt.figure(figsize=(6,6))
    for point, word in zip(transform , words):
        plt.scatter(point[0], point[1], c='r' if first else 'g')
        plt.annotate(
            word, 
            xy = (point[0], point[1]),
            xytext = (-7, -6) if first else (7, -6),
            textcoords = 'offset points',
            ha = 'right' if first else 'left',
            va = 'bottom',
            size = "medium"
            )
        first = not first 
    plt.title(name)
    plt.tight_layout()
plt.show()


#####################################################################################

'''
def word_averaging(wv, words):
    all_words, mean = set(), []
    
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(wv.layer_size,)

    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

def  word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, review) for review in text_list ])



X_train_word_average = word_averaging_list(wv,train_tokenized)
X_test_word_average = word_averaging_list(wv,test_tokenized)
'''
