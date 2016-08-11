from __future__ import division

'''
after: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb
'''

from collections import defaultdict
from collections import namedtuple
from collections import OrderedDict
from contextlib import contextmanager
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from random import sample
from random import shuffle
from sklearn import linear_model
from timeit import default_timer
import datetime
import gensim
import gensim.models.doc2vec
import io
import multiprocessing
import numpy as np
import random
import statsmodels.api as sm
import time 


models_exist = True

SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')

alldocs = []  # will hold all docs in original order
with io.open('data/aclImdb/alldata-id.txt', encoding='utf-8') as alldata:
    for line_no, line in enumerate(alldata):
        tokens = gensim.utils.to_unicode(line).split()
        words = tokens[1:]
        tags = [line_no] # `tags = [tokens[0]]` would also work at extra memory cost
        split = ['train','test','extra','extra'][line_no//25000]  # 25k train, 25k test, 25k extra
        sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//12500] # [12.5K pos, 12.5K neg]*2 then unknown
        alldocs.append(SentimentDocument(words, tags, split, sentiment))

train_docs = [doc for doc in alldocs if doc.split == 'train']
test_docs = [doc for doc in alldocs if doc.split == 'test']
doc_list = alldocs[:]  # for reshuffling per pass

print('%d docs: %d train-sentiment, %d test-sentiment' % (len(doc_list), len(train_docs), len(test_docs)))

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start
    
def logistic_predictor_from_data(train_targets, train_regressors):
    logit = sm.Logit(train_targets, train_regressors)
    predictor = logit.fit(disp=0)
    #print(predictor.summary())
    return predictor
  
def logistic_predictor_sklearn(train_targets, train_regressors):
    logistic_model = linear_model.LogisticRegression()
    logistic_model.fit(train_regressors, train_targets) 
    return logistic_model

def error_rate_for_model(test_model, train_set, test_set, infer=False, infer_steps=3, infer_alpha=0.1, infer_subsample=0.1):
    """Report error rate on test_doc sentiments, using supplied model and train_docs"""

    train_targets, train_regressors = zip(*[(doc.sentiment, test_model.docvecs[doc.tags[0]]) for doc in train_set])
    predictor = logistic_predictor_from_data(train_targets, sm.add_constant(train_regressors))
    predictor_sklearn = logistic_predictor_sklearn(train_targets, train_regressors)
    
    test_data = test_set
    if infer:
        if infer_subsample < 1.0:
            test_data = sample(test_data, int(infer_subsample * len(test_data)))
        test_regressors = [test_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha) for doc in test_data]
    else:
        test_regressors = [test_model.docvecs[doc.tags[0]] for doc in test_docs]
     
    # predict & evaluate
    test_predictions = predictor.predict(sm.add_constant(test_regressors))
    corrects = sum(np.rint(test_predictions) == [doc.sentiment for doc in test_data])
    errors = len(test_predictions) - corrects
    error_rate = float(errors) / len(test_predictions)
    
    test_predictions_sklearn = predictor_sklearn.predict(test_regressors)
    corrects_sklearn = sum(np.rint(test_predictions_sklearn) == [doc.sentiment for doc in test_data])
    errors_sklearn = len(test_predictions_sklearn) - corrects_sklearn
    error_rate_sklearn = float(errors_sklearn) / len(test_predictions_sklearn)
    
    return (error_rate, errors, error_rate_sklearn, errors_sklearn)

####################################################################################################

if not models_exist:

    cores = multiprocessing.cpu_count() / 2
    assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

    '''
    From the original notebook:

    Parameter choices below vary:

        100-dimensional vectors, as the 400d vectors of the paper don't seem to offer much benefit on this task
        similarly, frequent word subsampling seems to decrease sentiment-prediction accuracy, so it's left out
        cbow=0 means skip-gram which is equivalent to the paper's 'PV-DBOW' mode, matched in gensim with dm=0
        added to that DBOW model are two DM models, one which averages context vectors (dm_mean) and one which concatenates them (dm_concat, resulting in a much larger, slower, more data-hungry model)
        a min_count=2 saves quite a bit of model memory, discarding only words that appear in a single doc (and are thus no more expressive than the unique-to-each doc vectors themselves)
    '''

    simple_models = [
        # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
        Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=cores),
        # PV-DBOW 
        Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=cores),
        # PV-DM w/average
        Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores),
    ]

    # speed setup by sharing results of 1st model's vocabulary scan
    simple_models[0].build_vocab(alldocs)  # PV-DM/concat requires one special NULL word so it serves as template


    print(simple_models[0])
    for model in simple_models[1:]:
        model.reset_from(simple_models[0])
        print(model)


    models_by_name = OrderedDict((str(model), model) for model in simple_models)

    models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
    models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])


    best_error = defaultdict(lambda :1.0)  # to selectively-print only best errors achieved
    best_error_sklearn = defaultdict(lambda :1.0)

    alpha, min_alpha, passes = (0.025, 0.001, 20)
    alpha_delta = (alpha - min_alpha) / passes

    print("START %s" % datetime.datetime.now())

    for epoch in range(passes):
        shuffle(doc_list)  # shuffling gets best results
        
        for name, train_model in models_by_name.items():
            # train
            duration = 'na'
            train_model.alpha, train_model.min_alpha = alpha, alpha
            with elapsed_timer() as elapsed:
                train_model.train(doc_list)
                duration = '%.1f' % elapsed()
                
            # evaluate
            eval_duration = ''
            with elapsed_timer() as eval_elapsed:
                err, err_count, err_sklearn, err_count_sklearn = error_rate_for_model(train_model, train_docs, test_docs)
            eval_duration = '%.1f' % eval_elapsed()
            best_indicator = ' '
            if err <= best_error[name]:
                best_error[name] = err
                best_indicator = '*' 
            if err_sklearn <= best_error_sklearn[name]:
                best_error_sklearn[name] = err_sklearn
                
            print("%s%f [sklearn: %f]: %i passes : %s %ss %ss" % (best_indicator, err, err_sklearn, epoch + 1, name, duration, eval_duration))

            if ((epoch + 1) % 5) == 0 or epoch == 0:
                eval_duration = ''
                with elapsed_timer() as eval_elapsed:
                    infer_err, err_count, infer_err_sklearn, err_count_sklearn = error_rate_for_model(train_model, train_docs, test_docs, infer=True)
                eval_duration = '%.1f' % eval_elapsed()
                best_indicator = ' '
                if infer_err < best_error[name + '_inferred']:
                    best_error[name + '_inferred'] = infer_err
                    best_indicator = '*'
                if infer_err_sklearn < best_error_sklearn[name + '_inferred']:
                    best_error_sklearn[name + '_inferred'] = infer_err_sklearn
                    
                print("%s%f [sklearn: %f]: %i passes : %s %ss %ss" % (best_indicator, infer_err, infer_err_sklearn, epoch + 1, name + '_inferred', duration, eval_duration))

        print('completed pass %i at alpha %f' % (epoch + 1, alpha))
        alpha -= alpha_delta
        
    print("END %s" % str(datetime.datetime.now()))

    # print best error rates achieved
    for rate, name in sorted((rate, name) for name, rate in best_error.items()):
        print("%f %s" % (rate, name))
    print    
    for rate, name in sorted((rate, name) for name, rate in best_error_sklearn.items()):
        print("%f %s" % (rate, name))
        
    '''
    0.101600 dbow+dmm_inferred
    0.101920 dbow+dmm
    0.102400 Doc2Vec(dbow,d100,n5,mc2,t4)_inferred
    0.103600 dbow+dmc_inferred
    0.104120 dbow+dmc
    0.104280 Doc2Vec(dbow,d100,n5,mc2,t4)
    0.134280 Doc2Vec(dm/m,d100,n5,w10,mc2,t4)
    0.182520 Doc2Vec(dm/c,d100,n5,w5,mc2,t4)
    0.189600 Doc2Vec(dm/m,d100,n5,w10,mc2,t4)_inferred
    0.204000 Doc2Vec(dm/c,d100,n5,w5,mc2,t4)_inferred

    0.101600 dbow+dmm_inferred
    0.102080 dbow+dmm
    0.103200 Doc2Vec(dbow,d100,n5,mc2,t4)_inferred
    0.103600 dbow+dmc_inferred
    0.104160 dbow+dmc
    0.104400 Doc2Vec(dbow,d100,n5,mc2,t4)
    0.134120 Doc2Vec(dm/m,d100,n5,w10,mc2,t4)
    0.182560 Doc2Vec(dm/c,d100,n5,w5,mc2,t4)
    0.188800 Doc2Vec(dm/m,d100,n5,w10,mc2,t4)_inferred
    0.204000 Doc2Vec(dm/c,d100,n5,w5,mc2,t4)_inferred
    '''



    filenames = ['dmc', 'cbow', 'dmm']

    for model, filename in zip(models_by_name.values()[:3], filenames):
        model.save(filename)

      
#####################################################################################

if models_exist: 
    fnames = ['dmc', 'cbow', 'dmm']
    models = [Doc2Vec.load(fname) for fname in fnames]
else: 
    models = simple_models

doc_id = np.random.randint(models[0].docvecs.count)  # pick random doc, re-run cell for more examples
model = random.choice(models)  # and a random model
sims = model.docvecs.most_similar(doc_id, topn=model.docvecs.count)  # get *all* similar documents
print(u'TARGET (%d): %s\n' % (doc_id, ' '.join(alldocs[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: %s\n' % (label, sims[index], ' '.join(alldocs[sims[index][0]].words)))

   
for word in ['awful', 'awesome']:  
    print '\n\nSimilar words to: {}\n'.format(word)  
    for model in models:
        similar = model.most_similar(word, topn=10)
        print 'Model: {}\n{}\n'.format(model, similar)

for model in models:  
    print 'Model: {}: {}\n'.format(model, model.most_similar(positive=['awesome'], negative=['awful']))

for model in models:
    sections = model.accuracy('questions-words.txt')
    correct, incorrect = len(sections[-1]['correct']), len(sections[-1]['incorrect'])
    print('%s: %0.2f%% correct (%d of %d)' % (model, float(correct*100)/(correct+incorrect), correct, correct+incorrect))
  

  
  
