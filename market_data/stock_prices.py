import yfinance as yf

class StockPriceFetcher:
    """
    Fetch real-time and historical stock prices using yfinance.
    """
    def __init__(self):
        pass

    def get_current_price(self, ticker: str):
        """
        Fetch the latest price for the given ticker.
        """
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if not data.empty:
            return float(data['Close'].iloc[-1])
        return None

    def get_historical_prices(self, ticker: str, period: str = "1mo", interval: str = "1d"):
        """
        Fetch historical prices for the given ticker.
        period: e.g., '1mo', '3mo', '1y', 'max'
        interval: e.g., '1d', '1h', '5m'
        Returns a list of closing prices.
        """
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        if not data.empty:
            return list(data['Close'])
        return []
