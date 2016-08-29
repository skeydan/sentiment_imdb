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

with io.open('data/aclImdb/train-pos.txt', encoding='utf-8') as f: train_pos = pd.DataFrame({'review': list(f)})    
with io.open('data/aclImdb/train-neg.txt', encoding='utf-8') as f: train_neg = pd.DataFrame({'review': list(f)}) 
train_reviews = pd.concat([train_neg, train_pos], ignore_index=True)

with io.open('data/aclImdb/test-pos.txt', encoding='utf-8') as f: test_pos = pd.DataFrame({'review': list(f)})
with io.open('data/aclImdb/test-neg.txt', encoding='utf-8') as f: test_neg = pd.DataFrame({'review': list(f)})    
test_reviews = pd.concat([test_neg, test_pos], ignore_index=True)

   
'''
Not needed: input files are already cleaned up by preprocess_doc2vec.y

train_reviews_clean = []
for i in xrange( 0, train_reviews["review"].size ):
    train_reviews_clean.append(" ".join(preprocess.cleanup(train_reviews["review"][i], remove_stopwords=False)))

test_reviews_clean = []
for i in xrange( 0, test_reviews["review"].size ):
    test_reviews_clean.append(" ".join(preprocess.cleanup(test_reviews["review"][i], remove_stopwords=False)))
'''

X_train = train_reviews['review']
X_test = test_reviews['review']

y_train = np.append(np.zeros(12500), np.ones(12500))
y_test = np.append(np.zeros(12500), np.ones(12500)) 

stopwords_nltk = set(stopwords.words("english"))
relevant_words = set(['not', 'nor', 'no', 'wasn', 'ain', 'aren', 'very', 'only', 'but', 'don', 'isn', 'weren'])
stopwords_filtered = list(stopwords_nltk.difference(relevant_words))


print( '''
/******************************************************************************
*    Inspect vocabularies built by CountVectorizer for ngram ranges 1,2,3     *
******************************************************************************/
''')

for remove_stop_words in [stopwords_filtered, None]:
    print('\n\nStop words removed: {}\n*******************************'.format(remove_stop_words))
    for i in range(1,5):
        vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,
                                    stop_words = remove_stop_words, max_features = 10000, ngram_range = (1,i))

        # sparse matrix
        words_array = vectorizer.fit_transform(X_train).toarray()

        vocabulary = vectorizer.get_feature_names()

        counts = np.sum(words_array, axis=0)
        word_counts_overall = pd.DataFrame({'word': vocabulary, 'count': counts})
           
        word_counts_for_max_ngram = word_counts_overall[word_counts_overall.word.apply(lambda c: len(c.split()) >= i)]
           
        word_counts_for_max_ngram_sorted = word_counts_for_max_ngram.sort_values(by='count', ascending=False)
        print('\nMost frequent ngrams for ngrams in range 1 - {}:'.format(i))
            
        print(word_counts_for_max_ngram_sorted[:40])
        if remove_stop_words != None:
            filename = 'word_counts_sorted_ngram_' + str(i) + '_stopwords_removed.csv'
            word_counts_for_max_ngram_sorted.to_csv(filename)


print( '''
/******************************************************************************
*         Logistic Regression, using different numbers of ngrams              *
******************************************************************************/
''')

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,
                             stop_words = None, max_features = 10000)

logistic_model = LogisticRegression() 

logistic_pipeline = Pipeline([("vectorizer", vectorizer), ("logistic", logistic_model)])

search_params = dict(vectorizer__ngram_range = [(1,1), (1,2), (1,3)],
                     vectorizer__stop_words = [stopwords_filtered, None],
                     logistic__C = [0.01, 0.03, 0.05, 0.1])

best_logistic = GridSearchCV(logistic_pipeline, param_grid=search_params, cv=5, verbose=1)
best_logistic.fit(X_train, y_train)

print(best_logistic.best_params_)
print(best_logistic.grid_scores_)
print(best_logistic.best_estimator_.named_steps['logistic'].C)

utils.assess_classification_performance(best_logistic,  X_train, y_train, X_test, y_test)

'''
{'logistic__C': 0.03, 'vectorizer__ngram_range': (1, 2), 'vectorizer__stop_words': [u'all', u'just', u'being', u'over', u'through', u'yourselves', u'its', u'before', u'hadn', u'with', u'll', u'had', u'should', u'to', u'won', u'under', u'ours', u'has', u'wouldn', u'them', u'his', u'they', u'during', u'now', u'him', u'd', u'did', u'didn', u'these', u't', u'each', u'where', u'because', u'doing', u'theirs', u'some', u'we', u'are', u'our', u'ourselves', u'out', u'what', u'for', u'below', u're', u'does', u'above', u'between', u'mustn', u'she', u'be', u'hasn', u'after', u'here', u'shouldn', u'hers', u'by', u'on', u'about', u'couldn', u'of', u'against', u's', u'or', u'own', u'into', u'yourself', u'down', u'mightn', u'your', u'from', u'her', u'whom', u'there', u'been', u'few', u'too', u'then', u'themselves', u'was', u'until', u'more', u'himself', u'both', u'herself', u'than', u'those', u'he', u'me', u'myself', u'ma', u'this', u'up', u'will', u'while', u'can', u'were', u'my', u'at', u'and', u've', u'do', u'is', u'in', u'am', u'it', u'doesn', u'an', u'as', u'itself', u'o', u'have', u'further', u'their', u'if', u'again', u'that', u'when', u'same', u'any', u'how', u'other', u'which', u'you', u'shan', u'needn', u'haven', u'who', u'most', u'such', u'why', u'a', u'off', u'i', u'm', u'having', u'so', u'y', u'the', u'yours', u'once']}

Classification performance overview:
************************************
accuracy (train/test): 0.92876 / 0.88452

Confusion_matrix (training data):
[[11514   986]
 [  795 11705]]
Confusion_matrix (test data):
[[11003  1497]
 [ 1390 11110]]

'''

print( '''
/******************************************************************************
*         Logistic Regression, bigrams, stopwords filtered, C=0.03              *
******************************************************************************/
''')

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,
                             stop_words = stopwords_filtered, max_features = 10000, ngram_range = (1,2))
words_array = vectorizer.fit_transform(X_train).toarray()

logistic_model = LogisticRegression(C=0.03) 
logistic_model.fit(words_array, y_train)

vocabulary = vectorizer.get_feature_names()
coefs = logistic_model.coef_
word_importances = pd.DataFrame({'word': vocabulary, 'coef': coefs.tolist()[0]})
word_importances_sorted = word_importances.sort_values(by='coef', ascending = False)
print(word_importances_sorted)

word_importances_bigrams = word_importances_sorted[word_importances_sorted.word.apply(lambda c: len(c.split()) >= 2)]
print (word_importances_bigrams)


print( '''
/******************************************************************************
*                         SVM, using different numbers of ngrams              *
******************************************************************************/
''')

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,
                             stop_words = None, max_features = 10000)

svc_model = SVC() 

svc_pipeline = Pipeline([("vectorizer", vectorizer), ("svc", svc_model)])

search_params = dict(vectorizer__ngram_range = [(1,1), (1,2), (1,3)],
                    vectorizer__stop_words = [stopwords_filtered, None])

best_svc = GridSearchCV(svc_pipeline, param_grid=search_params, cv=5, verbose=1)
best_svc.fit(X_train, y_train)

print(best_svc.best_params_)
print(best_svc.grid_scores_)

utils.assess_classification_performance(best_svc,  X_train, y_train, X_test, y_test)



print( '''
/******************************************************************************
*         Random Forest Pipeline, using different numbers of ngrams           *
******************************************************************************/
''')

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,
                             stop_words = None, max_features = 10000)

forest_model = RandomForestClassifier(n_estimators = 100) 

forest_pipeline = Pipeline([("vectorizer", vectorizer), ("forest", forest_model)])

search_params = dict(vectorizer__ngram_range = [(1,1), (1,2), (1,3)],
                     vectorizer__stop_words = [stopwords_filtered, None],
                     forest__max_depth = [15, 20, 25])

best_forest = GridSearchCV(forest_pipeline, param_grid=search_params, cv=5, verbose=1)
best_forest.fit(X_train, y_train)

print(best_forest.best_params_)
print(best_forest.grid_scores_)
print( best_forest.best_estimator_.named_steps['forest'].feature_importances_)

utils.assess_classification_performance(best_forest,  X_train, y_train, X_test, y_test)



print( '''
/******************************************************************************
*                        QDA, using different numbers of ngrams               *
******************************************************************************/
''')

for i in range(1,4):
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,
                                stop_words = stopwords_filtered, max_features = 5000, ngram_range = (1,i))

    X_train_array = vectorizer.fit_transform(X_train).toarray()
    X_test_array = vectorizer.transform(X_test).toarray()

    qda_model = QuadraticDiscriminantAnalysis() 
    qda_model.fit(X_train_array, y_train)

    utils.assess_classification_performance(qda_model,  X_train_array, y_train, X_test_array, y_test)







