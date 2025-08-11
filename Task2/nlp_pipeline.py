# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 16:42:47 2025

@author: Acer
"""

import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('punkt_tab') 
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation & lowercase
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
    return tokens

def build_dataframe(raw_text):
    tokens = preprocess(raw_text)
    tagged = nltk.pos_tag(tokens)
    df = pd.DataFrame(tagged, columns=['Token', 'POS'])
    df['Sentiment'] = df['Token'].apply(lambda x: sia.polarity_scores(x)['compound'])
    return df

def get_freq_dist(tokens):
    return FreqDist(tokens)

def get_collocations(tokens, top_n=10):
    finder = BigramCollocationFinder.from_words(tokens)
    return finder.nbest(BigramAssocMeasures.likelihood_ratio, top_n)

def compute_sentiment_scores(text):
    return sia.polarity_scores(text)
