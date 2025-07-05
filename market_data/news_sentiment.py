import requests
from textblob import TextBlob

NEWSAPI_KEY = "YOUR_NEWSAPI_KEY"  # Replace with your NewsAPI key

class NewsSentimentFetcher:
    """
    Fetch and analyze news/social sentiment using NewsAPI and TextBlob.
    """
    def __init__(self, api_key=NEWSAPI_KEY):
        self.api_key = api_key

    def fetch_and_analyze(self, ticker: str, n_headlines: int = 5):
        """
        Fetch recent news headlines for the ticker and return average sentiment polarity.
        """
        url = (
            f"https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&language=en&pageSize={n_headlines}&apiKey={self.api_key}"
        )
        try:
            response = requests.get(url)
            response.raise_for_status()
            articles = response.json().get("articles", [])
            headlines = [a["title"] for a in articles if "title" in a]
            if not headlines:
                return 0.0
            sentiments = [TextBlob(h).sentiment.polarity for h in headlines]
            return sum(sentiments) / len(sentiments)
        except Exception as e:
            print(f"Error fetching news or analyzing sentiment: {e}")
            return 0.0
