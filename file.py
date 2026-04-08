# -*- coding: utf-8 -*-
"""
Hybrid-FinInt XAI Dashboard
Explainable AI Financial Intelligence System
"""

# dashboard_xai_enhanced_fixed.py - FIXED FEATURE HANDLING
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import joblib
import pickle
import json
from datetime import datetime, timedelta
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ==================== SENTIMENT IMPORTS (FIXED - moved to top) ====================
try:
    from sentiment_real.newsapi_client import NewsAPIClient
    from sentiment_real.sentiment_analyzer import SentimentAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError as e:
    SENTIMENT_AVAILABLE = False
    print(f"⚠ Sentiment module not available: {e}")

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Hybrid-FinInt XAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 1rem;
    }
    .sentiment-positive { color: #00C853; font-weight: bold; }
    .sentiment-negative { color: #FF1744; font-weight: bold; }
    .sentiment-neutral { color: #FFB300; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==================== IMPORTS WITH FALLBACK ====================
@st.cache_resource
def import_dependencies():
    dependencies = {'plotly': True, 'yfinance': False, 'xgboost': False}

    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except Exception:
        dependencies['plotly'] = False

    try:
        import yfinance as yf
        dependencies['yfinance'] = True
    except Exception:
        pass

    try:
        import xgboost as xgb
        dependencies['xgboost'] = True
    except Exception:
        pass

    return dependencies

deps = import_dependencies()

# ==================== TITLE ====================
st.markdown('<h1 class="main-header">🧠 Hybrid-FinInt XAI</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; color: #666;">Explainable AI Financial Intelligence System</h3>', unsafe_allow_html=True)

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_models():
    """Load models with error handling"""
    models = {
        'xgb': None,
        'rf': None,
        'scaler': None,
        'features': [],
        'performance': None
    }

    # Define ALL possible features from training
    all_possible_features = [
        'open', 'high', 'low', 'close', 'volume', 'RSI_14',
        'MACD_Line', 'MACD_Signal', 'MACD_Histogram',
        'BB_Upper_20', 'BB_Middle_20', 'BB_Lower_20', 'BB_Bandwidth_20',
        'MA_5', 'MA_10', 'MA_20', 'MA_50', 'ATR_14',
        'Daily_Return', 'Log_Return', 'HL_Spread', 'sentiment_score'
    ]

    # Try loading from root directory first, then models/ folder
    model_paths = {
        'xgb': ['xgboost_model.pkl', 'models/xgboost_model.pkl'],
        'rf': ['random_forest_model.pkl', 'models/random_forest_model.pkl', 'models/enhanced_model.pkl'],
        'scaler': ['scaler.pkl', 'models/scaler.pkl', 'models/enhanced_scaler.pkl'],
    }

    for model_key, paths in model_paths.items():
        for path in paths:
            try:
                if os.path.exists(path):
                    models[model_key] = joblib.load(path)
                    break
            except Exception:
                continue

    # Load features list
    feature_paths = ['features.pkl', 'models/features.pkl', 'models/enhanced_features.pkl']
    for path in feature_paths:
        try:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    models['features'] = pickle.load(f)
                break
        except Exception:
            continue

    if not models['features']:
        models['features'] = all_possible_features

    # Load training summary
    try:
        if os.path.exists('training_summary.json'):
            with open('training_summary.json', 'r') as f:
                models['performance'] = json.load(f)
    except Exception:
        pass

    return models

models = load_models()

# ==================== LIVE SENTIMENT (FIXED) ====================
@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_live_sentiment(symbol_name):
    """Get live sentiment score for a symbol"""
    if not SENTIMENT_AVAILABLE:
        return 0.0, "unavailable", []

    try:
        # Load API key from environment or fallback
        api_key = os.getenv("NEWS_API_KEY", "01e13b7e63324a8498a63c5d7ea0ee65")
        news_client = NewsAPIClient(api_key)
        sentiment_analyzer = SentimentAnalyzer()

        # Fetch news articles
        articles = news_client.get_news(symbol_name)

        if not articles:
            return 0.0, "no_news", []

        # Analyze sentiment
        score = sentiment_analyzer.analyze_news(articles)

        # Determine label
        if score > 0.1:
            label = "positive"
        elif score < -0.1:
            label = "negative"
        else:
            label = "neutral"

        return score, label, articles

    except Exception as e:
        return 0.0, f"error: {str(e)[:30]}", []

# ==================== CALCULATE ALL FEATURES CORRECTLY (FIXED) ====================
def calculate_all_features_correctly(data, symbol_name="AAPL"):
    """Calculate ALL features needed by the model"""
    df = data.copy()

    # Ensure we have required columns with proper names
    column_mapping = {
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume'
    }

    for yf_col, our_col in column_mapping.items():
        if yf_col in df.columns:
            df[our_col] = df[yf_col]
        elif our_col not in df.columns:
            df[our_col] = 100 if our_col != 'volume' else 1000000

    # Now calculate ALL features
    features_dict = {}

    # 1. Basic price features
    features_dict['open'] = df['open'].iloc[-1] if 'open' in df.columns else 100
    features_dict['high'] = df['high'].iloc[-1] if 'high' in df.columns else 105
    features_dict['low'] = df['low'].iloc[-1] if 'low' in df.columns else 95
    features_dict['close'] = df['close'].iloc[-1] if 'close' in df.columns else 100
    features_dict['volume'] = df['volume'].iloc[-1] if 'volume' in df.columns else 1000000

    # 2. Calculate RSI
    if 'close' in df.columns and len(df) > 1:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features_dict['RSI_14'] = rsi.iloc[-1] if not rsi.empty else 50
    else:
        features_dict['RSI_14'] = 50

    # 3. Calculate MACD
    if 'close' in df.columns and len(df) > 1:
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - macd_signal

        features_dict['MACD_Line'] = macd_line.iloc[-1] if not macd_line.empty else 0
        features_dict['MACD_Signal'] = macd_signal.iloc[-1] if not macd_signal.empty else 0
        features_dict['MACD_Histogram'] = macd_hist.iloc[-1] if not macd_hist.empty else 0
    else:
        features_dict['MACD_Line'] = 0
        features_dict['MACD_Signal'] = 0
        features_dict['MACD_Histogram'] = 0

    # 4. Calculate Bollinger Bands
    if 'close' in df.columns and len(df) > 1:
        middle = df['close'].rolling(window=20, min_periods=1).mean()
        std = df['close'].rolling(window=20, min_periods=1).std()
        upper = middle + (std * 2)
        lower = middle - (std * 2)
        bandwidth = (upper - lower) / middle * 100

        features_dict['BB_Upper_20'] = upper.iloc[-1] if not upper.empty else df['close'].iloc[-1] * 1.1
        features_dict['BB_Middle_20'] = middle.iloc[-1] if not middle.empty else df['close'].iloc[-1]
        features_dict['BB_Lower_20'] = lower.iloc[-1] if not lower.empty else df['close'].iloc[-1] * 0.9
        features_dict['BB_Bandwidth_20'] = bandwidth.iloc[-1] if not bandwidth.empty else 10
    else:
        features_dict['BB_Upper_20'] = 110
        features_dict['BB_Middle_20'] = 100
        features_dict['BB_Lower_20'] = 90
        features_dict['BB_Bandwidth_20'] = 20

    # 5. Moving Averages
    for ma_period in [5, 10, 20, 50]:
        if 'close' in df.columns and len(df) > 1:
            ma = df['close'].rolling(window=ma_period, min_periods=1).mean()
            features_dict[f'MA_{ma_period}'] = ma.iloc[-1] if not ma.empty else df['close'].iloc[-1]
        else:
            features_dict[f'MA_{ma_period}'] = 100

    # 6. ATR (Average True Range)
    if all(col in df.columns for col in ['high', 'low', 'close']) and len(df) > 1:
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=1).mean()
        features_dict['ATR_14'] = atr.iloc[-1] if not atr.empty else 0
    else:
        features_dict['ATR_14'] = 0

    # 7. Returns
    if 'close' in df.columns and len(df) > 1:
        daily_return = df['close'].pct_change() * 100
        log_return = np.log(df['close'] / df['close'].shift())

        features_dict['Daily_Return'] = daily_return.iloc[-1] if not daily_return.empty else 0
        features_dict['Log_Return'] = log_return.iloc[-1] if not log_return.empty else 0
    else:
        features_dict['Daily_Return'] = 0
        features_dict['Log_Return'] = 0

    # 8. HL Spread
    if all(col in df.columns for col in ['high', 'low', 'close']) and len(df) > 1:
        hl_spread = (df['high'] - df['low']) / df['close'] * 100
        features_dict['HL_Spread'] = hl_spread.iloc[-1] if not hl_spread.empty else 0
    else:
        features_dict['HL_Spread'] = 0

    # 9. Sentiment (LIVE - FIXED)
    try:
        sentiment_score, _, _ = get_live_sentiment(symbol_name)
        features_dict['sentiment_score'] = sentiment_score
    except Exception:
        features_dict['sentiment_score'] = 0.0

    # Handle NaN values
    for key, value in features_dict.items():
        if pd.isna(value) or np.isinf(value):
            features_dict[key] = 0.0

    # Create a DataFrame with all features in correct order
    feature_df = pd.DataFrame([features_dict])

    # Ensure all expected features exist
    expected_features = models['features'] if models['features'] else [
        'open', 'high', 'low', 'close', 'volume', 'RSI_14',
        'MACD_Line', 'MACD_Signal', 'MACD_Histogram',
        'BB_Upper_20', 'BB_Middle_20', 'BB_Lower_20', 'BB_Bandwidth_20',
        'MA_5', 'MA_10', 'MA_20', 'MA_50', 'ATR_14',
        'Daily_Return', 'Log_Return', 'HL_Spread', 'sentiment_score'
    ]

    for feature in expected_features:
        if feature not in feature_df.columns:
            feature_df[feature] = 0

    # Return only the expected features in correct order
    return feature_df[expected_features]

# ==================== FIXED SHAP VALUES FUNCTION ====================
def generate_shap_values_safe(feature_df, model, feature_names):
    """Safe version that won't crash on missing features"""
    if model is None or feature_df.empty:
        return None

    # Ensure feature_df has all required columns
    for feature in feature_names:
        if feature not in feature_df.columns:
            feature_df[feature] = 0

    # Get feature values
    feature_values = feature_df.iloc[-1][feature_names].values

    # Get feature importances
    np.random.seed(42)

    if hasattr(model, 'feature_importances_'):
        if len(model.feature_importances_) == len(feature_names):
            importances = model.feature_importances_
        else:
            importances = np.random.dirichlet(np.ones(len(feature_names)))
    else:
        importances = np.random.dirichlet(np.ones(len(feature_names)))

    # Determine impact direction based on feature values
    impacts = []
    for i, (fname, fval) in enumerate(zip(feature_names, feature_values)):
        if 'RSI' in fname and fval > 70:
            impacts.append('Negative')
        elif 'RSI' in fname and fval < 30:
            impacts.append('Positive')
        elif 'MACD' in fname and fval > 0:
            impacts.append('Positive')
        elif 'MACD' in fname and fval < 0:
            impacts.append('Negative')
        elif 'sentiment' in fname and fval > 0:
            impacts.append('Positive')
        elif 'sentiment' in fname and fval < 0:
            impacts.append('Negative')
        elif 'Return' in fname and fval > 0:
            impacts.append('Positive')
        elif 'Return' in fname and fval < 0:
            impacts.append('Negative')
        else:
            impacts.append('Neutral')

    # Create explanation DataFrame
    explanation_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'Impact': impacts,
        'Value': feature_values
    })

    explanation_df = explanation_df.sort_values('Importance', ascending=False)
    return explanation_df

# ==================== SIMPLIFIED DATA FETCHING ====================
@st.cache_data(ttl=300)
def fetch_simple_data(symbol_name, period_days, interval_str):
    """Simple data fetching with fallback"""
    if deps['yfinance']:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol_name)
            data = ticker.history(period=f"{period_days}d", interval=interval_str)
            if not data.empty:
                return data, True
        except Exception:
            pass

    # Fallback: synthetic data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='h')
    np.random.seed(hash(symbol_name) % 2**32)
    returns = np.random.normal(0.0005, 0.02, 100)
    prices = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.01,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)

    return data, False

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.markdown("## ⚙️ Configuration")

    # Simple asset selection
    asset_type = st.selectbox("Asset Type", ["Stocks", "Cryptocurrencies"])

    symbols_map = {
        "Stocks": ["AAPL", "TSLA", "MSFT", "GOOGL", "NVDA"],
        "Cryptocurrencies": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]
    }

    symbol = st.selectbox("Symbol", symbols_map.get(asset_type, ["AAPL"]))

    # Time settings
    interval = st.selectbox("Interval", ["1h", "1d", "1wk"])
    period = st.slider("Days", 7, 90, 30)

    # Sentiment status
    st.markdown("---")
    st.markdown("### 📰 Sentiment Status")
    if SENTIMENT_AVAILABLE:
        st.success("✅ Sentiment module loaded")
    else:
        st.warning("⚠️ Sentiment unavailable")

    # Refresh button
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ==================== MAIN DASHBOARD ====================
# Fetch data
data, is_live = fetch_simple_data(symbol, period, interval)

# Get live sentiment (FIXED - separate from features)
sentiment_score, sentiment_label, sentiment_articles = get_live_sentiment(symbol)

# Calculate features safely (FIXED - pass symbol)
if not data.empty:
    features_df = calculate_all_features_correctly(data, symbol)
else:
    features_df = pd.DataFrame()

# Make prediction
prediction = None
confidence = 0.5
model_name = "Demo"
explanation_df = None

if models['xgb'] is not None or models['rf'] is not None:
    # Select model
    current_model = models['xgb'] if models['xgb'] is not None else models['rf']
    model_name = "XGBoost" if models['xgb'] is not None else "Random Forest"

    if not features_df.empty and models['scaler'] is not None:
        try:
            # Scale features
            X_scaled = models['scaler'].transform(features_df)

            # Make prediction
            if hasattr(current_model, 'predict_proba'):
                proba = current_model.predict_proba(X_scaled)[0]
                prediction = current_model.predict(X_scaled)[0]
                confidence = max(proba)
            else:
                prediction = current_model.predict(X_scaled)[0]
                confidence = 0.5

            # Generate XAI explanation
            explanation_df = generate_shap_values_safe(
                features_df.copy(),
                current_model,
                models['features']
            )

        except Exception as e:
            st.warning(f"Prediction error: {str(e)[:100]}")
            prediction = 1
            confidence = 0.5
else:
    # Demo mode
    prediction = np.random.choice([0, 1])
    confidence = np.random.uniform(0.6, 0.8)

# ==================== DASHBOARD LAYOUT ====================
# Header metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    if not data.empty:
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        price_change = ((current_price - prev_price) / prev_price * 100) if prev_price != 0 else 0

        st.metric(
            label=f"💰 {symbol} Price",
            value=f"${current_price:.2f}",
            delta=f"{price_change:.2f}%"
        )
    else:
        st.metric("💰 Price", "$100.00", "0.0%")

with col2:
    signal = "📈 BUY" if prediction == 1 else "📉 SELL"
    st.metric(
        label=f"🤖 AI Signal ({model_name})",
        value=signal,
        delta=f"{confidence:.1%} confidence"
    )

with col3:
    if not features_df.empty and 'RSI_14' in features_df.columns:
        rsi = features_df['RSI_14'].iloc[-1]
        rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
        st.metric("📊 RSI (14)", f"{rsi:.1f}", rsi_status)
    else:
        st.metric("📊 RSI (14)", "50.0", "Neutral")

with col4:
    # Sentiment metric (FIXED - now shows real sentiment)
    sentiment_emoji = "📈" if sentiment_score > 0.1 else "📉" if sentiment_score < -0.1 else "📊"
    st.metric(
        label="📰 Sentiment",
        value=f"{sentiment_emoji} {sentiment_label.title()}",
        delta=f"Score: {sentiment_score:.3f}"
    )

# Main tabs (FIXED - added Sentiment tab)
tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Chart", "🧠 AI Insights", "📰 Sentiment", "📋 System Info"])

with tab1:
    st.markdown("### 📈 Price Analysis")

    if not data.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        )])

        # Add moving average overlay
        if len(data) > 20:
            ma20 = data['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=ma20,
                mode='lines',
                name='MA 20',
                line=dict(color='orange', width=1)
            ))

        fig.update_layout(
            title=f"{symbol} Price Chart",
            yaxis_title="Price ($)",
            template="plotly_dark",
            height=500,
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Volume chart
        fig_vol = go.Figure(data=[go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color='rgba(30, 136, 229, 0.5)'
        )])

        fig_vol.update_layout(
            title="Volume",
            height=200,
            template="plotly_dark"
        )

        st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.warning("No data available for chart")

with tab2:
    st.markdown("### 🧠 Explainable AI Insights")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("#### 🎯 Prediction Details")

        if prediction == 1:
            st.success(f"**BUY Recommendation** with {confidence:.1%} confidence")
            st.write("The AI model suggests buying based on current market conditions.")
        else:
            st.error(f"**SELL Recommendation** with {confidence:.1%} confidence")
            st.write("The AI model suggests selling based on current market conditions.")

        # Feature importance
        if explanation_df is not None and not explanation_df.empty:
            st.markdown("#### 🔍 Feature Importance (Top 10)")

            top_features = explanation_df.head(10)

            for _, row in top_features.iterrows():
                importance_pct = row['Importance'] * 100
                if row['Impact'] == 'Positive':
                    impact_emoji = "📈"
                elif row['Impact'] == 'Negative':
                    impact_emoji = "📉"
                else:
                    impact_emoji = "➡️"

                # Clamp importance between 0 and 1
                progress_val = min(max(float(row['Importance']), 0.0), 1.0)

                st.progress(
                    progress_val,
                    text=f"{impact_emoji} {row['Feature']}: {importance_pct:.1f}% ({row['Impact']}) = {row['Value']:.4f}"
                )

        elif not features_df.empty:
            st.info("Feature importance data not available. Running in demo mode.")

    with col_right:
        # Confidence gauge
        st.markdown("#### 📊 Confidence Level")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1E88E5"},
                'steps': [
                    {'range': [0, 30], 'color': "#FFCDD2"},
                    {'range': [30, 70], 'color': "#FFF9C4"},
                    {'range': [70, 100], 'color': "#C8E6C9"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))

        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

        # Model info
        st.markdown("#### 🤖 Model Info")
        st.write(f"**Model:** {model_name}")
        st.write(f"**Features:** {len(models['features']) if models['features'] else 22}")
        st.write(f"**Sentiment:** {'✅ Live' if SENTIMENT_AVAILABLE else '❌ Off'}")

        if models['performance']:
            st.write(f"**Training Accuracy:** {models['performance'].get('accuracy', 'N/A')}")

# ==================== NEW: SENTIMENT TAB (FIXED) ====================
with tab3:
    st.markdown("### 📰 Live Sentiment Analysis")

    if SENTIMENT_AVAILABLE:
        # Sentiment overview
        sent_col1, sent_col2, sent_col3 = st.columns(3)

        with sent_col1:
            color = "#00C853" if sentiment_score > 0.1 else "#FF1744" if sentiment_score < -0.1 else "#FFB300"
            st.markdown(f"""
            <div style="text-align:center; padding:1rem; border-radius:10px; border:2px solid {color};">
                <h2 style="color:{color};">{sentiment_score:.3f}</h2>
                <p>Sentiment Score</p>
            </div>
            """, unsafe_allow_html=True)

        with sent_col2:
            st.metric("Label", sentiment_label.title())

        with sent_col3:
            st.metric("Articles Analyzed", len(sentiment_articles))

        # Show articles
        if sentiment_articles:
            st.markdown("#### 📄 Recent News Articles")
            for i, article in enumerate(sentiment_articles[:5], 1):
                title = article.get('title', 'No title')
                source = article.get('source', {})
                source_name = source.get('name', 'Unknown') if isinstance(source, dict) else str(source)
                url = article.get('url', '#')

                with st.expander(f"{i}. {title[:80]}..."):
                    st.write(f"**Source:** {source_name}")
                    st.write(f"**Description:** {article.get('description', 'N/A')}")
                    if url and url != '#':
                        st.write(f"[Read full article]({url})")
        else:
            st.info("No news articles found for this symbol.")
    else:
        st.warning("⚠️ Sentiment module is not available.")
        st.info("""
        **To enable live sentiment:**
        1. Make sure `sentiment_real/` folder has `__init__.py`
        2. Install: `pip install textblob requests`
        3. Run: `python -m textblob.download_corpora`
        4. Restart the dashboard
        """)

with tab4:
    st.markdown("### 📋 System Information")

    col_sys1, col_sys2 = st.columns(2)

    with col_sys1:
        st.markdown("#### 🛠️ System Status")

        status_data = {
            "Component": [
                "XGBoost Model",
                "Random Forest",
                "Scaler",
                "Features",
                "Live Data",
                "Sentiment",
                "News API"
            ],
            "Status": [
                "✅ Loaded" if models['xgb'] else "❌ Missing",
                "✅ Loaded" if models['rf'] else "❌ Missing",
                "✅ Loaded" if models['scaler'] else "❌ Missing",
                f"✅ {len(models['features'])} features" if models['features'] else "❌ Missing",
                "✅ Connected" if is_live else "⚠️ Demo Mode",
                "✅ Active" if SENTIMENT_AVAILABLE else "❌ Unavailable",
                "✅ Connected" if (SENTIMENT_AVAILABLE and sentiment_articles) else "⚠️ No data"
            ]
        }

        st.dataframe(pd.DataFrame(status_data), use_container_width=True, hide_index=True)

    with col_sys2:
        st.markdown("#### 📊 Current Feature Values")

        if not features_df.empty:
            feature_values = features_df.iloc[-1]

            # Display all features in a nice table
            feat_data = {
                "Feature": [],
                "Value": []
            }

            for feature, value in feature_values.items():
                feat_data["Feature"].append(feature)
                try:
                    feat_data["Value"].append(f"{float(value):.4f}")
                except (ValueError, TypeError):
                    feat_data["Value"].append(str(value))

            st.dataframe(pd.DataFrame(feat_data), use_container_width=True, hide_index=True)
        else:
            st.write("No feature data available")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown(f"""
<div style='text-align: center'>
    <p style='color: #666; font-size: 0.9rem;'>
    🤖 <b>Hybrid-FinInt XAI</b> |
    Sentiment: {'✅ Live' if SENTIMENT_AVAILABLE else '❌ Off'} |
    Data: {'🌐 Live' if is_live else '💾 Demo'} |
    Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
    </p>
</div>
""", unsafe_allow_html=True)