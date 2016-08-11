from __future__ import division
import pandas as pd 
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import preprocess
import utils


reviews = pd.read_csv("data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

reviews_clean = []
for i in xrange( 0, reviews["review"].size ):
    reviews_clean.append(" ".join(preprocess.cleanup(reviews["review"][i], remove_stopwords=True)))



X_train, X_test, y_train, y_test = train_test_split(reviews_clean, reviews['sentiment'], test_size=0.2, random_state=0)


print '''
/******************************************************************************
*    Inspect vocabularies built by CountVectorizer for ngram ranges 1,2,3     *
******************************************************************************/
'''

for i in range(1,4):
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,
                                stop_words = None, max_features = 5000, ngram_range = (1,i))

    # sparse matrix
    words_array = vectorizer.fit_transform(X_train).toarray()

    vocabulary = vectorizer.get_feature_names()
    #print vocabulary[0:10]
    #print vectorizer.vocabulary_.get('able')

    counts = np.sum(words_array, axis=0)
    word_counts_overall = pd.DataFrame({'word': vocabulary, 'count': counts})
    
    word_counts_for_max_ngram = word_counts_overall[word_counts_overall.word.apply(lambda c: len(c.split()) >= i)]
    
    word_counts_for_max_ngram_sorted = word_counts_for_max_ngram.sort_values(by='count', ascending=False)
    print '\nMost frequent ngrams for ngrams in range 1 - {}:'.format(i)
    
    print word_counts_for_max_ngram_sorted[:40]


print '''
/******************************************************************************
*         Logistic Regression, using different numbers of ngrams              *
******************************************************************************/
'''

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,
                             stop_words = None, max_features = 5000)

logistic_model = LogisticRegression() 

logistic_pipeline = Pipeline([("vectorizer", vectorizer), ("logistic", logistic_model)])

search_params = dict(vectorizer__ngram_range = [(1,1), (1,2), (1,3)],
                     logistic__C = [0.01, 0.03, 0.05, 0.1])

best_logistic = GridSearchCV(logistic_pipeline, param_grid=search_params, cv=5, verbose=2)
best_logistic.fit(X_train, y_train)

print(best_logistic.best_params_)
print(best_logistic.grid_scores_)
print best_logistic.best_estimator_.named_steps['logistic'].C

utils.assess_classification_performance(best_logistic,  X_train, y_train, X_test, y_test)

'''
{'logistic__C': 0.05, 'vectorizer__ngram_range': (1, 2)}

Classification performance overview:
*****************************
accuracy (train/test): 0.9357 / 0.8806

Confusion_matrix (training data):
[[9241  711]
 [ 575 9473]]
Confusion_matrix (test data):
[[2220  328]
 [ 269 2183]]

'''

print '''
/******************************************************************************
*                         SVM, using different numbers of ngrams              *
******************************************************************************/
'''

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,
                             stop_words = None, max_features = 5000)

svc_model = SVC() 

svc_pipeline = Pipeline([("vectorizer", vectorizer), ("svc", svc_model)])

search_params = dict(vectorizer__ngram_range = [(1,1), (1,2), (1,3)])

best_svc = GridSearchCV(svc_pipeline, param_grid=search_params, cv=5, verbose=2)
best_svc.fit(X_train, y_train)

print(best_svc.best_params_)
print(best_svc.grid_scores_)

utils.assess_classification_performance(best_svc,  X_train, y_train, X_test, y_test)

'''


{'vectorizer__ngram_range': (1, 2)}

Classification performance overview:
*****************************
accuracy (train/test): 0.85325 / 0.8496

'''



print '''
/******************************************************************************
*         Random Forest Pipeline, using different numbers of ngrams           *
******************************************************************************/
'''

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,
                             stop_words = None, max_features = 5000)

forest_model = RandomForestClassifier(n_estimators = 100) 

forest_pipeline = Pipeline([("vectorizer", vectorizer), ("forest", forest_model)])

search_params = dict(vectorizer__ngram_range = [(1,1), (1,2), (1,3)],
                     forest__max_depth = [10, 15, 20])

best_forest = GridSearchCV(forest_pipeline, param_grid=search_params, cv=5, verbose=2)
best_forest.fit(X_train, y_train)

print(best_forest.best_params_)
print(best_forest.grid_scores_)
print best_forest.best_estimator_.named_steps['forest'].feature_importances_

utils.assess_classification_performance(best_forest,  X_train, y_train, X_test, y_test)

'''
'vectorizer__ngram_range': (1, 1), 'forest__max_depth': 15}

accuracy (train/test): 0.89085 / 0.836

Confusion_matrix (training data):
[[8350 1602]
 [ 581 9467]]
Confusion_matrix (test data):
[[2013  535]
 [ 285 2167]]
'''


print '''
/******************************************************************************
*                        QDA, using different numbers of ngrams               *
******************************************************************************/
'''

for i in range(1,4):
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,
                                stop_words = None, max_features = 5000, ngram_range = (1,i))

    X_train_array = vectorizer.fit_transform(X_train).toarray()
    X_test_array = vectorizer.transform(X_test).toarray()

    qda_model = QuadraticDiscriminantAnalysis() 
    qda_model.fit(X_train_array, y_train)

    utils.assess_classification_performance(qda_model,  X_train_array, y_train, X_test_array, y_test)



'''
/home/key/python/anaconda2/lib/python2.7/site-packages/sklearn/discriminant_analysis.py:688: UserWarning: Variables are collinear
  warnings.warn("Variables are collinear")
  
Classification performance overview:
*****************************
accuracy (train/test): 0.56565 / 0.5296

Confusion_matrix (training data):
[[9947    5]
 [8682 1366]]
Confusion_matrix (test data):
[[2528   20]
 [2332  120]]
'''




'''

forest_model.fit( X_train, y_train)    


print('Feature importances: {}\n'.format(forest_model.feature_importances_))


https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words

http://deeplearning.net/tutorial/lstm.html

http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

http://cs.stanford.edu/~quocle/paragraph_vector.pdf

http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf
'''









''' 
   Preprocessing steps 

1. 

function normalize_text {
    awk '{print tolower($0);}' < $1 | sed -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/"/ " /g' \
    -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' -e 's/\?/ \? /g' \
    -e 's/\;/ \; /g' -e 's/\:/ \: /g' > $1-norm
  }

  export LC_ALL=C
  for j in train/pos train/neg test/pos test/neg train/unsup; do
    rm temp
    for i in `ls aclImdb/$j`; do cat aclImdb/$j/$i >> temp; awk 'BEGIN{print;}' >> temp; done
    normalize_text temp
    mv temp-norm aclImdb/$j/norm.txt
  done
  mv aclImdb/train/pos/norm.txt aclImdb/train-pos.txt
  mv aclImdb/train/neg/norm.txt aclImdb/train-neg.txt
  mv aclImdb/test/pos/norm.txt aclImdb/test-pos.txt
  mv aclImdb/test/neg/norm.txt aclImdb/test-neg.txt
  mv aclImdb/train/unsup/norm.txt aclImdb/train-unsup.txt

  cat aclImdb/train-pos.txt aclImdb/train-neg.txt aclImdb/test-pos.txt aclImdb/test-neg.txt aclImdb/train-unsup.txt > aclImdb/alldata.txt
  awk 'BEGIN{a=0;}{print "_*" a " " $0; a++;}' < aclImdb/alldata.txt > aclImdb/alldata-id.txt
fi



'''
