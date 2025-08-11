README.md

markdown
📈 Stock Price Data Cleaning & Logistic Regression Modeling

This project processes **5 years of stock price data**, cleans it, engineers features,  
and builds a **logistic regression model** to predict if a stock’s price will go up the next day.

🛠 Steps Performed

1. Load Data
   - Reads `stock_details_5_years.csv`
   - Parses dates and optimizes data types

2. Data Cleaning
   - Sorts data by `Company` and `Date`
   - Forward/backward fills missing OHLCV values within each company
   - Fills missing `Dividends` and `Stock Splits` with zero
   - Removes outliers using **IQR method** for `Close` prices

3. Feature Engineering
   - Calculates **7-day average trading volume**
   - Calculates **daily returns**
   - Creates **polynomial & interaction terms** for OHLC features
   - Encodes `Company` as a numeric code

4. Save Cleaned Data
   - Outputs cleaned dataset as `stock_prices_cleaned.csv`

5. Modeling
   - Defines binary target: `1` if next day’s return > 0, else `0`
   - Uses **Logistic Regression (statsmodels)** to predict target
   - Displays model summary, odds ratios, and confidence intervals

📂 Project Structure


.
├── data/
│   └── stock\_details\_5\_years.csv        # Raw dataset
├── Cleaned data/
│   └── stock\_prices\_cleaned.csv         # Cleaned dataset
├── stock\_cleaning.py                    # Data cleaning & feature engineering
├── stock\_model.py                       # Logistic regression modeling
└── README.md


🚀 Usage
1. Install dependencies
   bash
   pip install -r requirements.txt


2. Run data cleaning

   bash
   python stock_cleaning.py


3. Run modeling

   bash
   python stock_model.py

