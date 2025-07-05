class HoldingsDB:
    """
    Simple in-memory holdings database (scaffold).
    """
    def __init__(self):
        self.holdings = {}

    def add(self, ticker: str, shares: float):
        self.holdings[ticker] = self.holdings.get(ticker, 0) + shares

    def remove(self, ticker: str, shares: float):
        if ticker in self.holdings:
            self.holdings[ticker] -= shares
            if self.holdings[ticker] <= 0:
                del self.holdings[ticker]

    def get_all(self):
        return self.holdings.copy()
