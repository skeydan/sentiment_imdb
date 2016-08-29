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

import utils

model_exists = True

with io.open('data/aclImdb/train-pos.txt', encoding='utf-8') as f: train_pos = pd.DataFrame({'review': list(f)})    
with io.open('data/aclImdb/train-neg.txt', encoding='utf-8') as f: train_neg = pd.DataFrame({'review': list(f)}) 
train_reviews = pd.concat([train_neg, train_pos], ignore_index=True)

with io.open('data/aclImdb/test-pos.txt', encoding='utf-8') as f: test_pos = pd.DataFrame({'review': list(f)})
with io.open('data/aclImdb/test-neg.txt', encoding='utf-8') as f: test_neg = pd.DataFrame({'review': list(f)})    
test_reviews = pd.concat([test_neg, test_pos], ignore_index=True)

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
    
print({k: model.vocab[k] for k in model.vocab.keys()[:5]})
print(model.syn0.shape)
print(model['movie'])

model.similarity('awesome', 'awful')

for word in ['awful', 'awesome']:  
    print('\n\nSimilar words to: {}\n'.format(word))  
    similar = model.most_similar(word, topn=10)
    print('Model: {}\n{}\n'.format(model, similar))

print('Model: {}: {}\n'.format(model, model.most_similar(positive=['awesome'], negative=['awful'])))

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


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0.
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print("Review %d of %d" % (counter, len(reviews)))
       # 
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs


# ****************************************************************
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word
# removal.


words_per_review_train = []
for review in train_reviews["review"]:
    words = [w for w in review.lower().split() if not w in stopwords.words("english")]
    words_per_review_train.append(words)

trainDataVecs = getAvgFeatureVecs( words_per_review_train, model, num_features )

words_per_review_test = []
for review in test_reviews["review"]:
    words = [w for w in review.lower().split() if not w in stopwords.words("english")]
    words_per_review_test.append(words)

testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )

