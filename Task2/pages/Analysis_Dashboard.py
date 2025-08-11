# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 16:44:58 2025

@author: Acer
"""

import streamlit as st
import matplotlib.pyplot as plt
from nlp_pipeline import preprocess, get_freq_dist, get_collocations, compute_sentiment_scores

st.title("📈 Analysis Dashboard")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file:
    raw_text = uploaded_file.read().decode("utf-8")
    tokens = preprocess(raw_text)

    # Frequency Plot
    st.subheader("Top 20 Frequent Words")
    freq_dist = get_freq_dist(tokens)
    top_words = freq_dist.most_common(20)

    words, counts = zip(*top_words)
    fig, ax = plt.subplots()
    ax.bar(words, counts)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Collocations
    st.subheader("Top 10 Collocations")
    collocations = get_collocations(tokens)
    for bigram in collocations:
        st.write(" ".join(bigram))

    # Sentiment
    st.subheader("Overall Sentiment Score")
    sentiment = compute_sentiment_scores(raw_text)
    st.json(sentiment)
