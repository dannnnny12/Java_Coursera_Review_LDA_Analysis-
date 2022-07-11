import re
import requests
import nltk
import gensim
nltk.download('punkt')
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk_stopwords = nltk.corpus.stopwords.words('english')
newStpWord = ['game','Game']
nltk_stopwords.extend(newStpWord)
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import gensim.corpora as corpora
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from gensim import models
from pprint import pprint
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
    arr.append(word_tokenize(result_list[ite]))
#print(arr)

for n in range(len(arr)): #stop words removal
    for element in arr[n]:
        if element in nltk_stopwords:
            pass
        else:
            if TextBlob(element).sentiment.polarity>0.15:
                pos.append(element)
            elif TextBlob(element).sentiment.polarity<-0.15:
                neg.append(element)

            #stpwd_arr.append(element)
#print(stpwd_arr)
#print("positive words:"+str(pos))
#print("negative words:"+str(neg))

#Positive Reviews LDA Analysis
id2word_pos = corpora.Dictionary([pos])
#print(id2word)
texts = pos
corpus_pos = [id2word_pos.doc2bow(text.split()) for text in pos]
#print(corpus)
#print(corpus[:1][0][:30])
num_topics = 10
lda_model_pos = models.LdaModel(corpus=corpus_pos,id2word=id2word_pos,num_topics=num_topics)
topic_list_pos = lda_model_pos.print_topics(10)
print("10 positive topic distribution：\n")
for topic in topic_list_pos:
    print(topic)


#Negative Reviews LDA Analysis
id2word_neg = corpora.Dictionary([neg])
#print(id2word)
corpus_neg = [id2word_neg.doc2bow(text.split()) for text in neg]
lda_model_neg = models.LdaModel(corpus=corpus_neg,id2word=id2word_neg,num_topics=num_topics)
topic_list_neg = lda_model_neg.print_topics(10)
print("10 negative topic distribution：\n")
for topic in topic_list_neg:
    print(topic)