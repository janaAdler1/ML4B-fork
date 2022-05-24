import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas_profiling
import seaborn as sns
import json
import glob
import json_lines
from tqdm.notebook import tqdm
from pprint import pprint
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from pycaret.nlp import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""
##########Extract text of tweet and name of party
party = []
files = glob.glob('data/*.jl')
for file in tqdm(files):
    with json_lines.open(file) as jl:
        for item in jl:
            if 'response' in item:
                if 'data' in item['response']:
                    for i in item['response']['data']:
                        party.append({'Party': item['account_data']['Partei'], 'Tweet': i['text']})
                        break
df = pd.DataFrame(party, columns = ['Party', 'Tweet'])

###########remove some symbols and stopwords
count = 0
for tweet in df['Tweet']:
    tweet = tweet.replace('ä', 'ae')
    tweet = tweet.replace('ö', 'oe')
    tweet = tweet.replace('ü', 'ue')
    tweet = tweet.replace('Ä', 'Ae')
    tweet = tweet.replace('Ö', 'Oe')
    tweet = tweet.replace('Ü', 'Ue')
    tweet = tweet.replace('ß', 'ss')
    df['Tweet'][count] = tweet
    if tweet.startswith('RT'):
        text = tweet[3:]        
        df['Tweet'][count] = text
    count += 1
df['Tweet'] = df['Tweet'].map(lambda x : ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x).split()))
df['Tweet'] = df['Tweet'].map(lambda x: x.lower())
df['Tweet'] = df['Tweet'].map(lambda x: re.sub(r'[^\w\s]', '', x))
df['Tweet'] = df['Tweet'].map(lambda x : re.sub(r'[^\x00-\x7F]+',' ', x))
stop_words = stopwords.words('german')
df['Tweet'] = df['Tweet'].map(lambda x : ' '.join([w for w in x.split() if w not in stop_words]))

df.to_csv('test.csv', index=False, columns = ['Party', 'Tweet'])
###########remove NaN rows from Dataset
df = pd.read_csv('test.csv')
df1 = df.dropna(thresh=2)
df1.to_csv('test.csv', index=False, columns = ['Party', 'Tweet'])
"""
df = pd.read_csv('test.csv')
X = df['Tweet'].values.astype('U')
y = df['Party']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
my_tags = df['Party'].unique()
#Naive Bayes Classifier
nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)

nb_pred = nb.predict(X_test)
##Evaluation
print('accuracy %s' % accuracy_score(nb_pred, y_test))
print(classification_report(y_test, nb_pred, target_names=my_tags))
#Linear Support Vector Machine
sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=12, max_iter=5, tol=None)),
               ])
sgd.fit(X_train, y_train)
sgd_pred = sgd.predict(X_test)
##Evaluation
print('accuracy %s' % accuracy_score(sgd_pred, y_test))
print(classification_report(y_test, sgd_pred, target_names=my_tags))

#Logistik Regression
logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
                  ])
logreg.fit(X_train, y_train)
lg_pred = logreg.predict(X_test)
##Evaluation
print('accuracy %s' % accuracy_score(lg_pred, y_test))
print(classification_report(y_test, lg_pred, target_names=my_tags))

##########test
tweet = "Machen Deutschland frei".lower()
nb_pred = nb.predict([tweet])
nb_pred
sgd_pred = sgd.predict([tweet])
sgd_pred
lg_pred = logreg.predict([tweet])
lg_pred
