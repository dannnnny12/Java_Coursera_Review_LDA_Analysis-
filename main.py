import re
import requests
import nltk
import gensim
nltk.download('punkt')
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk_stopwords = nltk.corpus.stopwords.words('english')
newStpWord = ['game','Game', ',','–','!','"','#','$','%','&',')','*','+','&',"i'm","I'm","I've","i've"]
nltk_stopwords.extend(newStpWord)
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import gensim.corpora as corpora
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from gensim import models
from pprint import pprint
import pandas as pd
result_list = []
for page in range(1,4): #hetch page1 to 3
    response = requests.get("https://www.coursera.org/learn/java-programming/reviews?page="+str(page))

    soup = BeautifulSoup(response.text, "html.parser")

    results = soup.find_all("div", class_="rc-CML font-lg show-soft-breaks cml-cui")
    #print(results)
    

#stpwd_arr = []

    for result in results: #extraction
        #result = str(result)
        #word_tokenize(result)
        result_list.append(result.getText())
#print(result_list)
    
arr = []
pos = []
neg = []
for ite in range(len(result_list)): #tokenization
    arr.append(result_list[ite])
#print(arr)
#print(arr)

pos_tweets = [('I love this car'),
    ('This view is amazing'),
    ('I feel great this morning'),
    ('I am so excited about the concert'),
    ('He is my best friend')]
def txt_sentiment(txt): 
    '''Utility function to classify sentiment of passed tweet using textblob's sentiment method'''
    # create TextBlob object of passed tweet text 
    analysis = TextBlob(txt) 
    return analysis.sentiment
    # set sentiment 
    if analysis.sentiment.polarity > 0.15:
        return 'positive'
    elif analysis.sentiment.polarity <-0.15: 
        return 'negative'
    else: 
        return 'neutral'
test = pd.DataFrame(result_list)
test.columns = ["reviews"]
test["reviews"] = test["reviews"].str.lower()
test['review_withoutSTPWD'] = test['reviews'].apply(lambda x: ' '.join([word for word in x.split() if word not in (nltk_stopwords)]))
test['sentiment'] = test['review_withoutSTPWD'].apply(lambda x: txt_sentiment(' '.join(x)))
#pprint(test)
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
print(test)