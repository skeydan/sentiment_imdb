from __future__ import division
import logging

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

import nltk.data
from nltk.corpus import stopwords
from gensim.models import word2vec
import io

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.svm import SVC

import utils

model_exists = True

with io.open('data/aclImdb/train-pos.txt', encoding='utf-8') as f: train_pos = pd.DataFrame({'review': list(f)})    
with io.open('data/aclImdb/train-neg.txt', encoding='utf-8') as f: train_neg = pd.DataFrame({'review': list(f)}) 
train_reviews = pd.concat([train_neg, train_pos], ignore_index=True)

with io.open('data/aclImdb/test-pos.txt', encoding='utf-8') as f: test_pos = pd.DataFrame({'review': list(f)})
with io.open('data/aclImdb/test-neg.txt', encoding='utf-8') as f: test_neg = pd.DataFrame({'review': list(f)})    
test_reviews = pd.concat([test_neg, test_pos], ignore_index=True)

y_train = np.append(np.zeros(12500), np.ones(12500))
y_test = np.append(np.zeros(12500), np.ones(12500))

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


if not model_exists:

    sentences_train = []  
    for review in train_reviews["review"]:
        sentences_train += [s.split() for s in tokenizer.tokenize(review)]
      
    print(len(sentences_train))
    print(sentences_train[0])

    sentences_test = []  
    for review in test_reviews["review"]:
        sentences_test += [s.split() for s in tokenizer.tokenize(review)]
        
    print(len(sentences_test))
    print(sentences_test[0])

    all_sentences = sentences_train + sentences_test


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
    
print({k: model.vocab[k] for k in model.vocab.keys()})
print(model.syn0.shape)
print(model['movie'])

model.similarity('awesome', 'awful')

for word in ['awful', 'awesome']:  
    print('\n\nSimilar words to: {}\n'.format(word))  
    similar = model.most_similar(word, topn=10)
    print('Model: {}\n{}\n'.format(model, similar))

print('Model: {}: {}\n'.format(model, model.most_similar(positive=['awesome'], negative=['awful'])))

model.doesnt_match("good bad awful terrible".split())


'''
sections = model.accuracy('data/questions-words.txt')
correct, incorrect = len(sections[-1]['correct']), len(sections[-1]['incorrect'])
print('%s: %0.2f%% correct (%d of %d)' % (model, float(correct*100)/(correct+incorrect), correct, correct+incorrect))
'''

########################################################################

words = ['story', 'movie','plot', 'film', 'good', 'bad', 'awful', 'awesome', 'man', 'woman', 'like', 'actor', 'actress']
vectors = model.syn0

pca = PCA(n_components=2)
pca_2d = pca.fit_transform(vectors)


tsne = TSNE(n_components=2, random_state=0, verbose=10, init='pca')
tsne_2d = tsne.fit_transform(vectors)


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


print( '''
/******************************************************************************
*        Build average vectors for train and test reviews                     *
******************************************************************************/
''')

def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0.
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
       if counter%1000. == 0.:
           print("Review %d of %d" % (counter, len(reviews)))
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
       counter = counter + 1.
    return reviewFeatureVecs


num_features = 100 

words_per_review_train = []
for review in train_reviews["review"]:
    words = [w for w in review.lower().split() if not w in stopwords.words("english")]
    words_per_review_train.append(words)

trainDataVecs = getAvgFeatureVecs( words_per_review_train, model, num_features )

i=0
words_per_review_test = []
for review in test_reviews["review"]:
    words = [w for w in review.lower().split() if not w in stopwords.words("english")]
    words_per_review_test.append(words)
    if i%100. == 0.: print(i)
    i=i+1

testDataVecs = getAvgFeatureVecs( words_per_review_test, model, num_features )


print( '''
/******************************************************************************
*                          Logistic RegressionCV                              *
******************************************************************************/
''')

logistic_model = LogisticRegressionCV()
logistic_model.fit(trainDataVecs, y_train)

logistic_model.C_

utils.assess_classification_performance(logistic_model,  trainDataVecs, y_train, testDataVecs, y_test)
'''
Classification performance overview:
************************************
accuracy (train/test): 0.8394 / 0.8346

Confusion_matrix (training data):
[[10433  2067]
 [ 1948 10552]]
Confusion_matrix (test data):
[[10573  1927]
 [ 2208 10292]]
'''


print( '''
/******************************************************************************
*                         SVM                                                *
******************************************************************************/
''')

svc_model = SVC() 
svc_model.fit(trainDataVecs, y_train)
utils.assess_classification_performance(svc_model,  trainDataVecs, y_train, testDataVecs, y_test)

'''
Classification performance overview:
************************************
accuracy (train/test): 0.7012 / 0.69676

Confusion_matrix (training data):
[[9198 3302]
 [4168 8332]]
Confusion_matrix (test data):
[[9360 3140]
 [4441 8059]]

'''

print( '''
/******************************************************************************
*                      Random Forest                                          *
******************************************************************************/
''')

forest_model = RandomForestClassifier(n_estimators = 100) 
forest_model.fit(trainDataVecs, y_train)
utils.assess_classification_performance(forest_model,  trainDataVecs, y_train, testDataVecs, y_test)

'''
Classification performance overview:
************************************
accuracy (train/test): 1.0 / 0.79048

Confusion_matrix (training data):
[[12500     0]
 [    0 12500]]
Confusion_matrix (test data):
[[9894 2606]
 [2632 9868]]

'''



print( '''
/******************************************************************************
*                        QDA                                                 *
******************************************************************************/
''')

qda_model = QuadraticDiscriminantAnalysis() 
qda_model.fit(trainDataVecs, y_train)
utils.assess_classification_performance(qda_model,  trainDataVecs, y_train, testDataVecs, y_test)

'''
Classification performance overview:
************************************
accuracy (train/test): 0.82652 / 0.79388

Confusion_matrix (training data):
[[10722  1778]
 [ 2559  9941]]
Confusion_matrix (test data):
[[10433  2067]
 [ 3086  9414]]
'''
