# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 16:44:39 2025

@author: Acer
"""

import streamlit as st
import pandas as pd
from nlp_pipeline import build_dataframe

st.title("📊 Data Explorer")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file:
    raw_text = uploaded_file.read().decode("utf-8")
    df = build_dataframe(raw_text)

    st.subheader("Sample DataFrame")
    st.dataframe(df.head(20))

    st.subheader("POS Distribution")
    pos_counts = df['POS'].value_counts()
    st.bar_chart(pos_counts)
