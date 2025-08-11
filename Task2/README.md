

README.md

markdown
# 📊 Streamlit NLP Dashboard

This project is a **two-page Streamlit application** for exploring and analyzing text data.  
It allows users to:
- Upload `.txt` files for analysis
- View top frequent words with bar plots
- Display top collocations (common bigrams)
- Compute sentiment scores
- Explore a DataFrame with tokenized text and POS tags

## 📂 Project Structure


.
├── app\_analysis.py        # Analysis Dashboard page
├── app\_explorer.py        # Data Explorer page
├── nlp\_pipeline.py        # NLP utility functions (preprocessing, sentiment, etc.)
├── requirements.txt       # Dependencies
└── README.md              # Project documentation

````

## 🚀 Usage
1. Install dependencies  
   bash
   pip install -r requirements.txt


2. Run the app
   For Analysis Dashboard:

   bash
   streamlit run app_analysis.py
   

   For Data Explorer:

   bash
   streamlit run app_explorer.py



