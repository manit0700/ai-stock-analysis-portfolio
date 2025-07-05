import yfinance as yf

class CompanyFundamentalsFetcher:
    """
    Fetch company fundamentals using yfinance.
    """
    def __init__(self):
        pass

    def fetch(self, ticker: str):
        """
        Fetches real company fundamentals for the given ticker.
        Returns a dictionary with pe_ratio, market_cap, dividend_yield, and more if available.
        """
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'ticker': ticker,
            'pe_ratio': info.get('trailingPE'),
            'market_cap': info.get('marketCap'),
            'dividend_yield': info.get('dividendYield'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'beta': info.get('beta'),
            '52_week_high': info.get('fiftyTwoWeekHigh'),
            '52_week_low': info.get('fiftyTwoWeekLow'),
        }
