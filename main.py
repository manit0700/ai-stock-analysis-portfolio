from ai_engine.price_forecast import PriceForecast
from ai_engine.sentiment_nlp import SentimentAnalyzer
from ai_engine.trading_strategy_generator import TradingStrategyGenerator
from portfolio.holdings_db import HoldingsDB
from portfolio.risk_metrics import calculate_volatility, calculate_sharpe_ratio
from market_data.stock_prices import StockPriceFetcher
from market_data.news_sentiment import NewsSentimentFetcher
from market_data.company_fundamentals import CompanyFundamentalsFetcher

if __name__ == "__main__":
    print("🚀 AI Stock Analysis and Investment Portfolio Backend")
    print("=" * 60)
    
    # Test Price Forecast
    print("\n📊 Testing Price Forecast...")
    pf = PriceForecast()
    print("Price Forecast (AAPL, 5 days):", pf.predict("AAPL", 5))

    # Test Sentiment Analysis
    print("\n📰 Testing Sentiment Analysis...")
    sa = SentimentAnalyzer()
    print("Sentiment (sample text):", sa.analyze("Apple stock is doing great!"))

    # Test Holdings DB
    print("\n💼 Testing Holdings Database...")
    db = HoldingsDB()
    db.add("AAPL", 10)
    db.add("GOOG", 5)
    db.remove("AAPL", 3)
    print("Holdings:", db.get_all())

    # Test Risk Metrics
    print("\n⚠️ Testing Risk Metrics...")
    prices = [150, 152, 148, 151, 153]
    returns = [0.01, -0.02, 0.015, 0.01, 0.005]
    print("Volatility:", calculate_volatility(prices))
    print("Sharpe Ratio:", calculate_sharpe_ratio(returns))

    # Test Stock Price Fetcher (real data)
    print("\n💰 Testing Real Stock Data...")
    spf = StockPriceFetcher()
    print("Current Stock Price (AAPL):", spf.get_current_price("AAPL"))
    print("Historical Prices (AAPL, 1mo):", len(spf.get_historical_prices("AAPL", period="1mo", interval="1d")), "data points")

    # Test News Sentiment Fetcher
    print("\n📰 Testing News Sentiment...")
    nsf = NewsSentimentFetcher()
    print("News Sentiment (AAPL):", nsf.fetch_and_analyze("AAPL"))

    # Test Company Fundamentals Fetcher
    print("\n🏢 Testing Company Fundamentals...")
    cff = CompanyFundamentalsFetcher()
    print("Company Fundamentals (AAPL):", cff.fetch("AAPL"))

    # Test AI Trading Strategy Generator
    print("\n🤖 Testing AI Trading Strategy Generator...")
    print("Note: This requires NewsAPI key for full functionality")
    strategy_gen = TradingStrategyGenerator()
    
    # Test with AAPL
    try:
        strategy = strategy_gen.generate_strategy("AAPL")
        if "error" not in strategy:
            print(f"✅ Strategy generated for AAPL:")
            print(f"   Recommendation: {strategy['recommendation']}")
            print(f"   Confidence: {strategy['confidence']}%")
            print(f"   Current Price: ${strategy['current_price']}")
            print(f"   Technical Signals: {len(strategy['technical_signals'])} signals")
        else:
            print(f"❌ Error: {strategy['error']}")
    except Exception as e:
        print(f"❌ Error testing strategy generator: {e}")

    print("\n✅ All tests completed!")
