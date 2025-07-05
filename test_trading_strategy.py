#!/usr/bin/env python3
"""
Test script for the AI Trading Strategy Generator
"""

from ai_engine.trading_strategy_generator import TradingStrategyGenerator
import json

def test_trading_strategy():
    """Test the trading strategy generator with real stock data"""
    
    # Initialize the strategy generator
    # Note: Replace "YOUR_NEWSAPI_KEY" with your actual NewsAPI key for news sentiment
    strategy_gen = TradingStrategyGenerator(news_api_key="YOUR_NEWSAPI_KEY")
    
    # Test stocks
    test_stocks = ["AAPL", "GOOGL", "TSLA", "MSFT", "AMZN"]
    
    print("🤖 AI Trading Strategy Generator")
    print("=" * 50)
    
    for ticker in test_stocks:
        print(f"\n📊 Analyzing {ticker}...")
        
        try:
            # Generate strategy
            strategy = strategy_gen.generate_strategy(ticker)
            
            if "error" in strategy:
                print(f"❌ Error: {strategy['error']}")
                continue
            
            # Display results
            print(f"💰 Current Price: ${strategy['current_price']}")
            print(f"🎯 Recommendation: {strategy['recommendation']}")
            print(f"📈 Confidence: {strategy['confidence']}%")
            print(f"📰 Sentiment Score: {strategy['sentiment_score']:.3f}")
            
            # Technical signals
            print("\n🔧 Technical Signals:")
            for signal_type, signal in strategy['technical_signals'].items():
                print(f"  {signal_type.upper()}: {signal}")
            
            # Risk metrics
            print(f"\n⚠️  Risk Metrics:")
            print(f"  Volatility: {strategy['risk_metrics']['volatility']:.3f}")
            print(f"  Sharpe Ratio: {strategy['risk_metrics']['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown: {strategy['risk_metrics']['max_drawdown']:.3f}")
            
            # Trading suggestions
            if strategy['recommendation'] != "HOLD":
                print(f"\n🎯 Trading Suggestions:")
                print(f"  Entry Points: ${strategy['entry_points']['primary']} / ${strategy['entry_points']['secondary']}")
                print(f"  Stop Loss: ${strategy['stop_loss']}")
                print(f"  Target Price: ${strategy['target_price']}")
            
            # News headlines (if available)
            if strategy['news_headlines']:
                print(f"\n📰 Recent News Headlines:")
                for i, headline in enumerate(strategy['news_headlines'][:3], 1):
                    print(f"  {i}. {headline}")
            
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Error analyzing {ticker}: {e}")
            continue

if __name__ == "__main__":
    test_trading_strategy() 