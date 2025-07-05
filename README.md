# AI Stock Analysis and Investment Portfolio Backend

This project provides the backend modules for an AI-powered stock analysis and investment portfolio management system. It includes:

- **AI Engine:** Price forecasting (LSTM, scaffold) and sentiment analysis (NLP, scaffold)
- **Portfolio Management:** Holdings database and risk metrics calculation
- **Market Data Layer:** Fetches real-time stock prices, news/social sentiment, and company fundamentals (scaffold)

## Structure

- `ai_engine/` — ML models for price forecasting and sentiment analysis
- `portfolio/` — Holdings database and risk metrics
- `market_data/` — Stock prices, news sentiment, and fundamentals fetchers
- `main.py` — Entry point for testing modules

## Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

## Next Steps
- Implement API endpoints (FastAPI)
- Integrate real data sources and models
- Build iOS frontend
