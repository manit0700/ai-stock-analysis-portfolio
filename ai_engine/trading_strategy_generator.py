import yfinance as yf
import pandas as pd
import numpy as np
from textblob import TextBlob
import requests
from datetime import datetime, timedelta
import talib

class TradingStrategyGenerator:
    """
    AI-powered trading strategy generator that combines:
    - Live stock data from yfinance
    - Real-time news sentiment analysis
    - Technical indicators
    - Pattern recognition
    """
    
    def __init__(self, news_api_key="YOUR_NEWSAPI_KEY"):
        self.news_api_key = news_api_key
        
    def get_live_stock_data(self, ticker: str, period: str = "1mo"):
        """Fetch live stock data with technical indicators"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if data.empty:
                return None
                
            # Calculate technical indicators
            data['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)
            data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
            data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
            data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['Close'])
            data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['Close'])
            
            return data
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return None
    
    def get_news_sentiment(self, ticker: str, days: int = 7):
        """Fetch and analyze news sentiment for the stock"""
        try:
            url = f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&language=en&pageSize=20&apiKey={self.news_api_key}"
            response = requests.get(url)
            response.raise_for_status()
            
            articles = response.json().get("articles", [])
            headlines = [a["title"] for a in articles if "title" in a]
            
            if not headlines:
                return 0.0, []
                
            sentiments = [TextBlob(h).sentiment.polarity for h in headlines]
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            return avg_sentiment, headlines[:5]  # Return top 5 headlines
        except Exception as e:
            print(f"Error fetching news sentiment: {e}")
            return 0.0, []
    
    def analyze_technical_signals(self, data):
        """Analyze technical indicators and generate signals"""
        if data is None or len(data) < 50:
            return {}
            
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        signals = {}
        
        # RSI Analysis
        if latest['RSI'] < 30:
            signals['rsi'] = 'oversold'
        elif latest['RSI'] > 70:
            signals['rsi'] = 'overbought'
        else:
            signals['rsi'] = 'neutral'
        
        # MACD Analysis
        if latest['MACD'] > latest['MACD_signal'] and prev['MACD'] <= prev['MACD_signal']:
            signals['macd'] = 'bullish_crossover'
        elif latest['MACD'] < latest['MACD_signal'] and prev['MACD'] >= prev['MACD_signal']:
            signals['macd'] = 'bearish_crossover'
        else:
            signals['macd'] = 'neutral'
        
        # Moving Average Analysis
        if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
            signals['ma'] = 'bullish_trend'
        elif latest['Close'] < latest['SMA_20'] < latest['SMA_50']:
            signals['ma'] = 'bearish_trend'
        else:
            signals['ma'] = 'mixed'
        
        # Bollinger Bands Analysis
        if latest['Close'] < latest['BB_lower']:
            signals['bb'] = 'oversold'
        elif latest['Close'] > latest['BB_upper']:
            signals['bb'] = 'overbought'
        else:
            signals['bb'] = 'neutral'
        
        return signals
    
    def generate_strategy(self, ticker: str):
        """Generate comprehensive trading strategy"""
        print(f"Analyzing {ticker}...")
        
        # Get live stock data
        data = self.get_live_stock_data(ticker)
        if data is None:
            return {"error": "Could not fetch stock data"}
        
        # Get news sentiment
        sentiment_score, headlines = self.get_news_sentiment(ticker)
        
        # Analyze technical signals
        technical_signals = self.analyze_technical_signals(data)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(technical_signals, sentiment_score)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(data)
        
        # Generate strategy
        strategy = {
            "ticker": ticker,
            "current_price": float(data['Close'].iloc[-1]),
            "recommendation": recommendation,
            "confidence": self._calculate_confidence(technical_signals, sentiment_score),
            "technical_signals": technical_signals,
            "sentiment_score": sentiment_score,
            "news_headlines": headlines,
            "risk_metrics": risk_metrics,
            "entry_points": self._suggest_entry_points(data, recommendation),
            "stop_loss": self._suggest_stop_loss(data, recommendation),
            "target_price": self._suggest_target_price(data, recommendation),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return strategy
    
    def _generate_recommendation(self, technical_signals, sentiment_score):
        """Generate buy/sell/hold recommendation"""
        bullish_signals = 0
        bearish_signals = 0
        
        # Count technical signals
        for signal_type, signal in technical_signals.items():
            if signal in ['oversold', 'bullish_crossover', 'bullish_trend']:
                bullish_signals += 1
            elif signal in ['overbought', 'bearish_crossover', 'bearish_trend']:
                bearish_signals += 1
        
        # Add sentiment influence
        if sentiment_score > 0.1:
            bullish_signals += 1
        elif sentiment_score < -0.1:
            bearish_signals += 1
        
        # Generate recommendation
        if bullish_signals > bearish_signals + 1:
            return "BUY"
        elif bearish_signals > bullish_signals + 1:
            return "SELL"
        else:
            return "HOLD"
    
    def _calculate_confidence(self, technical_signals, sentiment_score):
        """Calculate confidence level (0-100)"""
        signal_count = len(technical_signals)
        sentiment_confidence = abs(sentiment_score) * 50
        
        # Base confidence on signal strength
        base_confidence = min(signal_count * 15, 60)
        total_confidence = min(base_confidence + sentiment_confidence, 100)
        
        return round(total_confidence, 1)
    
    def _calculate_risk_metrics(self, data):
        """Calculate risk metrics"""
        returns = data['Close'].pct_change().dropna()
        
        return {
            "volatility": float(returns.std() * np.sqrt(252)),  # Annualized
            "sharpe_ratio": float((returns.mean() * 252) / (returns.std() * np.sqrt(252))),
            "max_drawdown": float((data['Close'] / data['Close'].expanding().max() - 1).min()),
            "beta": 1.0  # Placeholder - would need market data for real beta
        }
    
    def _suggest_entry_points(self, data, recommendation):
        """Suggest entry points based on recommendation"""
        current_price = data['Close'].iloc[-1]
        
        if recommendation == "BUY":
            return {
                "primary": round(current_price * 0.98, 2),  # 2% below current
                "secondary": round(current_price * 0.95, 2)  # 5% below current
            }
        elif recommendation == "SELL":
            return {
                "primary": round(current_price * 1.02, 2),  # 2% above current
                "secondary": round(current_price * 1.05, 2)  # 5% above current
            }
        else:
            return {"primary": current_price, "secondary": current_price}
    
    def _suggest_stop_loss(self, data, recommendation):
        """Suggest stop loss levels"""
        current_price = data['Close'].iloc[-1]
        
        if recommendation == "BUY":
            return round(current_price * 0.92, 2)  # 8% below current
        elif recommendation == "SELL":
            return round(current_price * 1.08, 2)  # 8% above current
        else:
            return None
    
    def _suggest_target_price(self, data, recommendation):
        """Suggest target price"""
        current_price = data['Close'].iloc[-1]
        
        if recommendation == "BUY":
            return round(current_price * 1.15, 2)  # 15% above current
        elif recommendation == "SELL":
            return round(current_price * 0.85, 2)  # 15% below current
        else:
            return current_price 