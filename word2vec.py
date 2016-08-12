from __future__ import division
import logging

import pandas as pd 
import numpy as np

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

# word2vec expects whole sentences
def review_to_sentences(review, tokenizer, remove_stopwords=False):
    review_sentences = tokenizer.tokenize(review.strip())
    review_sentences_clean = []
    for sentence in review_sentences:
        if len(sentence) > 0:
            review_sentences_clean.append(preprocess.cleanup(sentence))
    return review_sentences_clean
    
sentences_train = []  

i=0
for review in train_reviews["review"]:
    sentences_train += review_to_sentences(review, tokenizer)
    i = i+1
    if i % 1000 == 0: print 'Train sentences cleaned: {}'.format(i)  
    
print len(sentences_train)
print sentences_train[0:5]

sentences_test = []  

i=0
for review in test_reviews["review"]:
    sentences_test += review_to_sentences(review, tokenizer)
    i = i+1
    if i % 1000 == 0: print 'Test sentences cleaned: {}'.format(i)  
    
print len(sentences_test)
print sentences_test[0:5]


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 2       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

model = word2vec.Word2Vec(sentences_train, workers=num_workers, size=num_features, min_count = min_word_count,
                          window = context, sample = downsampling)

model.init_sims(replace=True)

model_name = "word2vec_300features_40minwords_10context"
model.save(model_name)

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



words = ['movie', 'film', 'good', 'bad', 'awful', 'awesome', 'man', 'woman', 'like', 'story', 'plot', 'actor', 'actress']
vectors = [model[word] for word in words]

pca = PCA(n_components=2, whiten=True)
pca_2d = pca.fit(vectors).transform(vectors)
   
tsne = TSNE(n_components=2, random_state=0)
tsne_2d = tsne.fit_transform(vectors)


for transform in [pca_2d, tsne_2d]:    
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
        first = not first if alternate else first
    plt.title('PCA')
    plt.tight_layout()
    plt.show()


