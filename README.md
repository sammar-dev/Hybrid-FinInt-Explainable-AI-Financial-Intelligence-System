Hybrid-FinInt: Explainable AI Financial Intelligence System

A hybrid machine learning pipeline that combines technical analysis, NLP-based sentiment analysis, and deep learning to predict financial market movements — for both stocks and cryptocurrencies.

 Table of Contents
Overview
Architecture
Project Structure
Pipeline
Models
Features
Installation
Usage
Dashboard
Performance
Roadmap
Dependencies
License
 Overview

Hybrid-FinInt is an end-to-end financial prediction system built in Python 3.11.

It:

Collects real-time market data via yfinance
Computes 20+ technical indicators
Adds news sentiment analysis (NLP)
Trains an ensemble of:
XGBoost
Random Forest
LSTM
Provides results via an interactive Streamlit XAI dashboard
📊 Supported Assets

Stocks

AAPL, TSLA, MSFT, GOOGL, META, NVDA, AMZN, NFLX


Crypto

BTC-USD, ETH-USD, SOL-USD, DOGE-USD, BNB-USD, XRP-USD, ADA-USD


 Architecture
Data Sources
 ├── yfinance (stocks + crypto)
 └── NewsAPI (news headlines)
        │
        ▼
Phase 1: Data Collection
        │
        ▼
Phase 2: Technical Indicators
        │
        ▼
Phase 3: Sentiment Analysis
        │
        ▼
Phase 5: Feature Engineering
        │
        ▼
Model Training
 ├── XGBoost
 ├── Random Forest
 └── LSTM
        │
        ▼
Streamlit XAI Dashboard
 Project Structure
Hybrid-FinInt/
│
├── run.py
├── Sammar.py
├── phase2_indicators.py
├── phase3_sentiment.py
├── phase5_enhance.py
├── prepare_for_colab.py
│
├── models/
│   ├── xgboost_model.pkl
│   ├── random_forest_model.pkl
│   ├── lstm_model.pth
│   ├── scaler.pkl
│   └── features.pkl
│
├── src/
│   ├── data_collection/
│   ├── technical/
│   ├── sentiment/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── colab/
│
├── prediction_graphs/
└── packages/
 Pipeline
🔹 Phase 1 — Data Collection
Fetches 30 days of 1-hour OHLCV data
Covers 15 assets
Output: data/raw/
🔹 Phase 2 — Technical Indicators

Adds:

RSI (14)
MACD (12/26/9)
Bollinger Bands
Moving Averages (5, 10, 20, 50)
ATR
Returns & spreads

Output: data/processed/with_indicators_*.csv

🔹 Phase 3 — Sentiment Analysis
Uses NewsAPI + TextBlob / FinBERT
Produces sentiment_score ∈ [-1, 1]
🔹 Phase 4 — Model Training

Performed in Google Colab

Models:

XGBoost
Random Forest
LSTM (PyTorch)

Target:
 Will price increase in next 1-hour?

🔹 Phase 5 — Feature Engineering

Adds:

Lag features (t-1, t-2, t-3)
Interaction features
Time-based features
Asset-type flags
Improved target (>0.5% move)
 Models
Model	File	Type	Notes
XGBoost	xgboost_model.pkl	Gradient Boosting	Primary
Random Forest	random_forest_model.pkl	Ensemble	200 trees
LSTM	lstm_model.pth	Deep Learning	Time-series
 Features

Base features (22):

OHLCV + RSI + MACD + Bollinger Bands + MA + ATR + Returns + Sentiment

Phase 5 adds:

Lagged features
Interaction features
Time features
 Installation
# Clone repo
git clone <your-repo-url>
cd Hybrid-FinInt

# Create environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy scipy yfinance scikit-learn \
xgboost torch joblib streamlit plotly \
textblob requests transformers
 NewsAPI (Optional)
Get key: https://newsapi.org
Add in:
self.news_api_key = "YOUR_KEY"
 Usage

Run pipeline step-by-step:

python run.py
python phase2_indicators.py
python phase3_sentiment.py
python prepare_for_colab.py
python phase5_enhance.py
🚀 Launch Dashboard
streamlit run Sammar.py
📊 Dashboard Features
📈 Candlestick charts (Plotly)
🤖 Live predictions
🔍 Feature importance (XAI)
📰 Sentiment overlays
📉 Model performance metrics
📈 Performance
Stage	Accuracy
Baseline	~50%
+ Sentiment (mock)	~52–53%
+ Real Sentiment	~55–60%
Future potential	~60–65%

Binary prediction task (50% = random baseline)

 Roadmap
 Twitter/Reddit sentiment integration
 Macro indicators (VIX, DXY, rates)
 Full FinBERT integration
 Backtesting module
 Docker deployment
 REST API for predictions
 Dependencies
Python 3.11
pandas, numpy, scipy
scikit-learn
xgboost
PyTorch
Streamlit
Plotly
TextBlob
Transformers
 License

This project is for educational purposes only.
It does not provide financial advice.
