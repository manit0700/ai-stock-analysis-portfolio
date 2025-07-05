from ai_engine.price_forecast import PriceForecast
from ai_engine.sentiment_nlp import SentimentAnalyzer
from portfolio.holdings_db import HoldingsDB
from portfolio.risk_metrics import calculate_volatility, calculate_sharpe_ratio
from market_data.stock_prices import StockPriceFetcher
from market_data.news_sentiment import NewsSentimentFetcher
from market_data.company_fundamentals import CompanyFundamentalsFetcher

if __name__ == "__main__":
    # Test Price Forecast
    pf = PriceForecast()
    print("Price Forecast (AAPL, 5 days):", pf.predict("AAPL", 5))

    # Test Sentiment Analysis
    sa = SentimentAnalyzer()
    print("Sentiment (sample text):", sa.analyze("Apple stock is doing great!"))

    # Test Holdings DB
    db = HoldingsDB()
    db.add("AAPL", 10)
    db.add("GOOG", 5)
    db.remove("AAPL", 3)
    print("Holdings:", db.get_all())

    # Test Risk Metrics
    prices = [150, 152, 148, 151, 153]
    returns = [0.01, -0.02, 0.015, 0.01, 0.005]
    print("Volatility:", calculate_volatility(prices))
    print("Sharpe Ratio:", calculate_sharpe_ratio(returns))

    # Test Stock Price Fetcher (real data)
    spf = StockPriceFetcher()
    print("Current Stock Price (AAPL):", spf.get_current_price("AAPL"))
    print("Historical Prices (AAPL, 1mo):", spf.get_historical_prices("AAPL", period="1mo", interval="1d"))

    # Test News Sentiment Fetcher
    nsf = NewsSentimentFetcher()
    print("News Sentiment (AAPL):", nsf.fetch_and_analyze("AAPL"))

    # Test Company Fundamentals Fetcher
    cff = CompanyFundamentalsFetcher()
    print("Company Fundamentals (AAPL):", cff.fetch("AAPL"))
