import yfinance as yf
import pandas as pd
import numpy as np

def fetch_market_data(holdings):
    tickers = list(holdings.keys())
    
    # 1. Download History
    # auto_adjust=True is implicit in new yfinance, so we grab 'Close'
    history = yf.download(tickers, period="1y")['Close']
    
    # 2. Download Benchmark (Handle failure gracefully)
    try:
        benchmark = yf.download("^GSPC", period="1y")['Close']
    except:
        benchmark = None
    
    fundamentals = []
    
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            info = t.info
            
            # Smart Price Logic
            price = info.get('currentPrice') or info.get('regularMarketPreviousClose') 
            if not price:
                try: price = history[ticker].iloc[-1]
                except: price = 0.0
            
            data = holdings.get(ticker, {})
            qty = data['qty'] if isinstance(data, dict) else data
            buy_price = data.get('buy_price', 0.0) if isinstance(data, dict) else 0.0
            
            val = price * qty
            
            fundamentals.append({
                "Ticker": ticker, "Quantity": qty, "Position Value": val,
                "Cost Basis": buy_price * qty, "Unrealized P&L": val - (buy_price * qty),
                "Sector": info.get("sector", "Unknown")
            })
        except Exception as e:
            print(f"Skipping {ticker}: {e}")
            
    return history, benchmark, pd.DataFrame(fundamentals), {}

def calculate_portfolio_metrics(history, benchmark):
    if isinstance(history, pd.Series): history = history.to_frame()
    returns = history.pct_change(fill_method=None).dropna()
    
    vol = returns.std() * np.sqrt(252)
    corr = returns.corr() if returns.shape[1] > 1 else pd.DataFrame([[1.0]], columns=returns.columns, index=returns.columns)
    div_score = int(max(0, min(100, (1 - corr.values.mean()) * 100)))
    
    # Benchmark Comparison (Fixed for 1D error)
    port_cum = (1 + returns.mean(axis=1)).cumprod()
    
    bench_cum = None
    if benchmark is not None:
        # FORCE 1D Series
        if isinstance(benchmark, pd.DataFrame):
            benchmark = benchmark.iloc[:, 0]
            
        bench_returns = benchmark.pct_change(fill_method=None).dropna()
        common_index = port_cum.index.intersection(bench_returns.index)
        bench_cum = (1 + bench_returns.loc[common_index]).cumprod()
        port_cum = port_cum.loc[common_index]

    data_map = {"Portfolio": port_cum}
    if bench_cum is not None: data_map["S&P 500"] = bench_cum
        
    comp_df = pd.DataFrame(data_map) * 100
    return vol, corr, div_score, comp_df

def run_monte_carlo(history, weights, current_val, days=90, sims=200):
    if isinstance(history, pd.Series): history = history.to_frame()
    daily_ret = history.pct_change(fill_method=None).dropna()
    
    # Weighted Returns
    port_ret = pd.Series(0.0, index=daily_ret.index)
    total_w = sum(weights.values())
    if total_w == 0: total_w = 1
    
    for t, w in weights.items():
        if t in daily_ret.columns: 
            port_ret += daily_ret[t] * (w / total_w)
            
    mu, sigma = port_ret.mean(), port_ret.std()
    
    # Fast Vectorized Simulation
    sim_data = {}
    for i in range(sims):
        shocks = np.random.normal(0, 1, days)
        daily_growth = np.exp((mu - 0.5 * sigma**2) + sigma * shocks)
        path = [current_val]
        for g in daily_growth: path.append(path[-1] * g)
        sim_data[f"Sim {i}"] = path
        
    return pd.DataFrame(sim_data)