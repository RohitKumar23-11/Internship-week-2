📚 Multi-Project Repository

This repository contains four major Python projects developed for data analysis, modeling, and benchmarking:

1. NumPy Neural Network for Regression – Fully connected neural network built from scratch for regression tasks.
2. Streamlit NLP Dashboard – Interactive text analysis and visualization.
3. Time-Series Benchmarking – Performance comparison of rolling operations using Pandas, NumPy, and Numba. 
4. Stock Price Data Cleaning & Logistic Regression Modeling – Data preprocessing, feature engineering, and predictive modeling on stock market data.

1️⃣ NumPy Neural Network for Regression

📌 Overview

Implements a fully connected feedforward neural network from scratch using NumPy:

* Configurable activation functions (`ReLU`, `Sigmoid`)
* Manual forward & backward propagation
* **SGD optimizer** with mini-batch training
* Mean Squared Error loss
* Synthetic cubic function regression task

📂 Structure


numpy_nn/
├── model.py              # NeuralNetwork and SGD classes
├── train.py              # Training script on synthetic data


🚀 Run

bash
python train.py


📦 Requirements

Install all dependencies with:

bash
pip install -r requirements.txt


requirements.txt

streamlit
matplotlib
pandas
nltk
numpy
numba
seaborn
scikit-learn
statsmodels


2️⃣ Streamlit NLP Dashboard

📌 Overview
A two-page Streamlit application for text analytics, allowing:
- Uploading `.txt` files
- Viewing top word frequencies
- Extracting collocations (bigrams)
- Computing sentiment scores
- POS tagging and distribution charts

📂 Structure


nlp\_dashboard/
├── app\_analysis.py       # Page 1: Frequency, collocations, sentiment
├── app\_explorer.py       # Page 2: POS tagging and DataFrame explorer
├── nlp\_pipeline.py       # NLP helper functions



🚀 Run
bash
pip install -r requirements.txt
streamlit run app_analysis.py
# OR
streamlit run app_explorer.py


3️⃣ Time-Series Benchmarking

📌 Overview

Benchmarks rolling mean/variance operations using:

* Pandas rolling
* NumPy stride tricks
* Numba JIT compilation

Generates:

* `results.csv` – benchmark timings
* `benchmark_plot.png` – log-log performance plot

📂 Structure


time_series_benchmark/
├── benchmark.py          # Main benchmarking script
├── timeseries_utils.py   # Rolling mean/variance, EWMA, FFT helpers


🚀 Run

bash
python benchmark.py


4️⃣ Stock Price Data Cleaning & Logistic Regression Modeling

📌 Overview

Processes 5 years of stock price data:

* Fills missing OHLCV data per company
* Removes outliers using IQR
* Generates rolling averages & daily returns
* Creates polynomial interaction features
* Encodes categorical companies
* Saves cleaned CSV

Then:

* Defines binary classification target: Will price go up next day?
* Trains **Logistic Regression (statsmodels)**
* Outputs odds ratios & confidence intervals

📂 Structure


stock_analysis/
├── stock_cleaning.py     # Data cleaning & feature engineering
├── stock_model.py        # Logistic regression modeling
├── data/                 # Raw CSV
└── Cleaned data/         # Processed CSV


🚀 Run

bash
python stock_cleaning.py
python stock_model.py


