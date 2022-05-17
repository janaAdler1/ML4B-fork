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
#from pycaret.classification import *

"""
#Extract text of tweet and name of party
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
print(df)
#remove some symbols and stopwords
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

df['Tweet'] = df['Tweet'].map(lambda x : ' '.join(re.s!ub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x).split()))
df['Tweet'] = df['Tweet'].map(lambda x: x.lower())
df['Tweet'] = df['Tweet'].map(lambda x: re.sub(r'[^\w\s]', '', x))
df['Tweet'] = df['Tweet'].map(lambda x : re.sub(r'[^\x00-\x7F]+',' ', x))
stop_words = stopwords.words('german')
df['Tweet'] = df['Tweet'].map(lambda x : ' '.join([w for w in x.split() if w not in stop_words]))
df.to_csv('test.csv', index=False, columns = ['Party', 'Tweet'])
"""
df = pd.read_csv('testq.csv')
df.head(n = 20)

