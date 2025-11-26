import yfinance as yf
import pandas as pd
import numpy as np

def fetch_market_data(holdings):
    """
    Fetches data and calculates position value based on user quantity and buy price.
    holdings: {'AAPL': {'qty': 10, 'buy_price': 150}}
    """
    tickers = list(holdings.keys())
    
    # 1. Download History
    history = yf.download(tickers, period="1y")['Close']
    
    # 2. Download Benchmark (S&P 500)
    benchmark = yf.download("^GSPC", period="1y")['Close']
    
    fundamentals = []
    news_cache = {}
    
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            info = t.info
            
            current_price = info.get('currentPrice') or info.get('regularMarketPreviousClose') or 0.0
            if current_price == 0.0 and not history.empty:
                 try:
                     current_price = history[ticker].iloc[-1]
                 except:
                     pass

            # Unpack data
            data = holdings.get(ticker, {})
            if isinstance(data, (int, float)):
                qty = data
                buy_price = 0.0
            else:
                qty = data.get('qty', 0)
                buy_price = data.get('buy_price', 0.0)
            
            position_value = current_price * qty
            cost_basis = buy_price * qty
            unrealized_pnl = position_value - cost_basis

            cap = info.get("marketCap", 0)
            if cap > 10_000_000_000:
                cap_category = "Large Cap"
            elif cap > 2_000_000_000:
                cap_category = "Mid Cap"
            else:
                cap_category = "Small Cap"

            fundamentals.append({
                "Ticker": ticker,
                "Quantity": qty,
                "Avg Buy Price": buy_price,
                "Current Price": current_price,
                "Position Value": position_value,
                "Cost Basis": cost_basis,
                "Unrealized P&L": unrealized_pnl,
                "Sector": info.get("sector", "Unknown"),
                "Cap Category": cap_category
            })
            news_cache[ticker] = t.news[:3]
            
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    return history, benchmark, pd.DataFrame(fundamentals), news_cache

def calculate_portfolio_metrics(history_df, benchmark_df):
    """Calculates Volatility, Correlation, Diversification AND Benchmark Comparison."""
    
    if isinstance(history_df, pd.Series):
        history_df = history_df.to_frame()
    
    returns = history_df.pct_change(fill_method=None).dropna()
    volatility = returns.std() * np.sqrt(252)
    
    if returns.shape[1] < 2:
        correlation = pd.DataFrame([[1.0]], index=returns.columns, columns=returns.columns)
    else:
        correlation = returns.corr()
        
    avg_corr = correlation.values.mean()
    div_score = max(0, min(100, (1 - avg_corr) * 100))
    
    # Performance Comparison Logic
    port_cum_return = (1 + returns.mean(axis=1)).cumprod() - 1
    
    if isinstance(benchmark_df, pd.DataFrame):
        benchmark_df = benchmark_df.iloc[:, 0]
    bench_returns = benchmark_df.pct_change(fill_method=None).dropna()
    bench_cum_return = (1 + bench_returns).cumprod() - 1
    
    common_dates = port_cum_return.index.intersection(bench_cum_return.index)
    comparison_df = pd.DataFrame({
        "My Portfolio": port_cum_return.loc[common_dates],
        "S&P 500 Benchmark": bench_cum_return.loc[common_dates]
    }) * 100
    
    return volatility, correlation, int(div_score), comparison_df

def run_monte_carlo(history_df, weights, current_value, days=90, simulations=200):
    """
    Simulates future portfolio value using Geometric Brownian Motion.
    """
    # 1. Calculate Portfolio Historical Daily Returns
    if isinstance(history_df, pd.Series):
        history_df = history_df.to_frame()
    
    daily_returns = history_df.pct_change(fill_method=None).dropna()
    
    # Weighted average return of the portfolio
    # weights is a dict like {'AAPL': 0.5, 'MSFT': 0.5}
    portfolio_daily_ret = pd.Series(0.0, index=daily_returns.index)
    
    total_weight = sum(weights.values())
    for ticker, weight in weights.items():
        if ticker in daily_returns.columns:
            norm_weight = weight / total_weight
            portfolio_daily_ret += daily_returns[ticker] * norm_weight
            
    # 2. Get Statistics (Mean and Volatility)
    mu = portfolio_daily_ret.mean()
    sigma = portfolio_daily_ret.std()
    
    # 3. Run Simulation
    # Formula: Price_t = Price_{t-1} * exp((mu - 0.5 * sigma^2) + sigma * Z)
    dt = 1 # 1 day step
    simulation_results = pd.DataFrame()
    
    for i in range(simulations):
        prices = [current_value]
        for d in range(days):
            shock = np.random.normal(0, 1) # Random shock (Z)
            drift = mu - (0.5 * sigma**2)
            change = drift + sigma * shock
            price = prices[-1] * np.exp(change)
            prices.append(price)
            
        simulation_results[f"Sim {i}"] = prices
        
    return simulation_results