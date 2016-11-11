import io
import pandas as pd
import numpy as np
from gensim.models import word2vec

pd.set_option('display.max_colwidth', 400)

with io.open('data/aclImdb/train-pos.txt', encoding='utf-8') as f:
    train_pos = pd.DataFrame({'review': list(f)})    
with io.open('data/aclImdb/train-neg.txt', encoding='utf-8') as f:
    train_neg = pd.DataFrame({'review': list(f)}) 
train_reviews = pd.concat([train_neg, train_pos], ignore_index=True)

with io.open('data/aclImdb/test-pos.txt', encoding='utf-8') as f:
    test_pos = pd.DataFrame({'review': list(f)})
with io.open('data/aclImdb/test-neg.txt', encoding='utf-8') as f:
    test_neg = pd.DataFrame({'review': list(f)})    
test_reviews = pd.concat([test_neg, test_pos], ignore_index=True)
  
X_train = train_reviews['review']
X_test = test_reviews['review']


# load the trained model from disk
model = word2vec.Word2Vec.load('models/word2vec_100features')
print(model.syn0.shape)


print( '''
/******************************************************************************
*                 Investigate similarity of awful and awesome                 *
******************************************************************************/
''')

def get_context(text, word, size):
    text_as_list = text.split()
    index = text_as_list.index(word)
    try:
        context_indices = range(index-size, index+size) 
        context = [text_as_list[i] for i in context_indices]
        return context
    except:    
        if size == 1: return None
        size = size/2
        get_context(text, word, size)

contains_awful = X_train[X_train.str.contains(' awful ')]
context_awful = contains_awful.apply(lambda x: get_context(x, 'awful', 8))
#print(context_awful)

contains_awesome = X_train[X_train.str.contains(' awesome ')]
context_awesome = contains_awesome.apply(lambda x: get_context(x, 'awesome', 8))
#print(context_awesome)

awesome_list = ['amazing', 'incredible', 'awful', 'excellent', 'fantastic', 'alright', 'cool', 'outstanding', 'astounding', 'terrific']
awful_list = ['terrible', 'horrible', 'atrocious', 'dreadful', 'appalling', 'horrendous', 'abysmal', 'amazing', 'awesome', 'bad']               

num_awesome = len(X_train[X_train.str.contains(' awesome ')])
print('num_awesome: {}'.format(num_awesome))
num_awful = len(X_train[X_train.str.contains(' awful ')])
print('num_awful: {}'.format(num_awful))

for word in awesome_list:
    contains_both = X_train[X_train.str.contains(' awesome ') & X_train.str.contains(word)]
    print('awesome and {}: {}'.format(word, len(contains_both)))
    
for word in awful_list:
    contains_both = X_train[X_train.str.contains(' awful ') & X_train.str.contains(word)]
    print('awful and {}: {}'.format(word, len(contains_both)))
    
    

                                     
                                     
