# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 12:04:32 2018

@author: user
"""

# Importing Libraries

import PyPDF2
import numpy as np
import pandas as pd
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#%%
## Opening PDF File from specified path
pdfFileObject=open('E:\Portfolio\JavaBasics-notes.pdf','rb')
pdfReader=PyPDF2.PdfFileReader(pdfFileObject)
total_pages=pdfReader.numPages
text=[]
for page in range(total_pages):
    pageObj=pdfReader.getPage(page)
    text.append(pageObj.extractText())
pdfFileObject.close()
#%%
## Applying Text normalization to content
text=str(text)
stop_words=stopwords.words('english')
words=word_tokenize(text)
table=str.maketrans('','',string.punctuation)
stripped=[w.translate(table) for w in words]
words=[w.lower() for w in stripped if w.isalpha()]
words=[w for w in words if w not in stop_words ]
#%%
## Applying POS(Parts-of-Speech) to content
POS_tag=nltk.pos_tag(words)
#%%
## Applying Lemmatization(part of Text normalization) to content
wordnet_lemmatizer=WordNetLemmatizer()
adj_tags=['JJ']
lemmatized_text=[]
for word in POS_tag:
    if word[1] in adj_tags:
        lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0],pos='a')))
    else:
        lemmatized_text.append(str(wordnet_lemmatizer.lemmatize(word[0])))
#%%
## Applying POS(Parts-of-Speech) to Lemmatized text
POS_tag=nltk.pos_tag(lemmatized_text)
print("Lemmatized text with POS Tags")
print(POS_tag[:50])

stopwords=[]
wanted_POS=['NN','NNS','NNP','NNPS','JJ','JJR','JJS','VBG','FW']
for word in POS_tag:
    if word[1] not in wanted_POS:
        stopwords.append(word[0])
        
words_filter=[w_f for w_f in lemmatized_text if not w_f in stopwords]
POS_filter=nltk.pos_tag(words_filter)
print(POS_filter[:50])

#%%
## Using TF-IDF and CountVectorizer for weightage of features
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
count_vec=CountVectorizer(stop_words='english')
tf_transformer=TfidfTransformer(use_idf=True)
counts=count_vec.fit_transform(words_filter)
tfidf=tf_transformer.fit_transform(counts)
#%%
xxx=zip(count_vec.get_feature_names(),tf_transformer.idf_)
list_feature_weight=list(xxx)
list_feature_weight.sort()
print('List of Features along with their weights',list_feature_weight[:50])
#%%
## Creating a dataframe and appending list in it
df=pd.DataFrame(list_feature_weight,columns=['Features','Weights'])
print('DataFrame featuring Features and Weights',df)