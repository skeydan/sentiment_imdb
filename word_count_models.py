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

import io
from nltk.corpus import stopwords

import preprocess
import utils


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

   
    
train_reviews_clean = []
for i in xrange( 0, train_reviews["review"].size ):
    train_reviews_clean.append(" ".join(preprocess.cleanup(train_reviews["review"][i], remove_stopwords=False)))

test_reviews_clean = []
for i in xrange( 0, test_reviews["review"].size ):
    test_reviews_clean.append(" ".join(preprocess.cleanup(test_reviews["review"][i], remove_stopwords=False)))

X_train = train_reviews_clean
X_test = test_reviews_clean

y_train = np.append(np.zeros(12500), np.ones(12500))
y_test = np.append(np.zeros(12500), np.ones(12500)) 

stopwords_nltk = stopwords.words("english")


print '''
/******************************************************************************
*    Inspect vocabularies built by CountVectorizer for ngram ranges 1,2,3     *
******************************************************************************/
'''

for remove_stop_words in [stopwords_nltk, None]:
    print '\n\nStop words removed: {}\n*******************************'.format(remove_stop_words)    
    for i in range(1,4):
        vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,
                                    stop_words = remove_stop_words, max_features = 5000, ngram_range = (1,i))

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
                     vectorizer__stop_words = [stopwords_nltk, None],
                     logistic__C = [0.01, 0.03, 0.05, 0.1])

best_logistic = GridSearchCV(logistic_pipeline, param_grid=search_params, cv=5, verbose=1)
best_logistic.fit(X_train, y_train)

print(best_logistic.best_params_)
print(best_logistic.grid_scores_)
print best_logistic.best_estimator_.named_steps['logistic'].C

utils.assess_classification_performance(best_logistic,  X_train, y_train, X_test, y_test)



print '''
/******************************************************************************
*                         SVM, using different numbers of ngrams              *
******************************************************************************/
'''

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,
                             stop_words = None, max_features = 5000)

svc_model = SVC() 

svc_pipeline = Pipeline([("vectorizer", vectorizer), ("svc", svc_model)])

search_params = dict(vectorizer__ngram_range = [(1,1), (1,2), (1,3)],
                    vectorizer__stop_words = [stopwords_nltk, None])

best_svc = GridSearchCV(svc_pipeline, param_grid=search_params, cv=5, verbose=1)
best_svc.fit(X_train, y_train)

print(best_svc.best_params_)
print(best_svc.grid_scores_)

utils.assess_classification_performance(best_svc,  X_train, y_train, X_test, y_test)



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
                     vectorizer__stop_words = [stopwords_nltk, None],
                     forest__max_depth = [15, 20, 25])

best_forest = GridSearchCV(forest_pipeline, param_grid=search_params, cv=5, verbose=1)
best_forest.fit(X_train, y_train)

print(best_forest.best_params_)
print(best_forest.grid_scores_)
print best_forest.best_estimator_.named_steps['forest'].feature_importances_

utils.assess_classification_performance(best_forest,  X_train, y_train, X_test, y_test)



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







