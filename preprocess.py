import re
from bs4 import BeautifulSoup  
import nltk
from nltk.corpus import stopwords

def cleanup(review, remove_stopwords=False):
  
    # Remove HTML
    #review_text = BeautifulSoup(review, 'lxml').get_text() 
    review_text = review.replace('<br />', ' ')
    #
    # Remove non-letters     
    # TBD: do NOT remove everything, keep emoticons etc
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # optionally remove stopwords
    if remove_stopwords:
      stops = set(stopwords.words("english"))                  
      words = [w for w in words if not w in stops]  
    
    # TBD: stemming (Porter or lemmatize)
    
    return words
