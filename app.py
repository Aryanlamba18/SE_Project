import streamlit as st
import pandas as pd
import plotly.express as px
from data_engine import fetch_market_data, calculate_portfolio_metrics, run_monte_carlo
# Import Agents
from agents import NewsAgent, RiskAgent, AdvisorAgent, RecommenderAgent

st.set_page_config(page_title="IntelliQuant Suite", layout="wide")
st.title("🤖 IntelliQuant: Agentic Finance Suite")
st.markdown("### Team 2 - Final Feasibility Prototype")

# --- INITIALIZE TEAM ---
if 'team' not in st.session_state:
    st.session_state.team = {
        'news': NewsAgent(),
        'risk': RiskAgent(),
        'advisor': AdvisorAgent(),
        'rec': RecommenderAgent()
    }

# --- SIDEBAR: PORTFOLIO INPUT ---
with st.sidebar:
    st.header("1. Your Portfolio")
    default_data = pd.DataFrame([
        {"Ticker": "AAPL", "Quantity": 10, "Avg Buy Price": 150.0},
        {"Ticker": "MSFT", "Quantity": 5, "Avg Buy Price": 300.0}
    ])
    edited_df = st.data_editor(default_data, num_rows="dynamic", key="editor")
    
    holdings = {}
    for index, row in edited_df.iterrows():
        if row["Ticker"] and row["Quantity"] > 0:
            holdings[row["Ticker"].strip().upper()] = {
                "qty": row["Quantity"],
                "buy_price": row.get("Avg Buy Price", 0)
            }
            
    run_btn = st.button("🚀 Run Analysis")

# --- MAIN APPLICATION ---
if run_btn and len(holdings) > 0:
    # 1. DATA LAYER
    with st.spinner("🤖 Data Agent working..."):
        history, benchmark, fundamentals, news = fetch_market_data(holdings)
        volatility, correlation, div_score, comparison_df = calculate_portfolio_metrics(history, benchmark)
        
        # Monte Carlo
        weights = dict(zip(fundamentals['Ticker'], fundamentals['Position Value']))
        total_val = fundamentals['Position Value'].sum()
        sim_df = run_monte_carlo(history, weights, total_val)

    # Context Variables
    total_cost = fundamentals['Cost Basis'].sum()
    total_pnl = total_val - total_cost
    pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
    
    advisor_context = f"""
    Portfolio Value: ${total_val:,.2f}
    Total P&L: ${total_pnl:,.2f} ({pnl_pct:.1f}%)
    Diversification Score: {div_score}/100
    Holdings: {list(holdings.keys())}
    """

    # --- TIER 1: ADVISOR SCORECARD (New Format) ---
    st.header("1. Chief Advisor Verdict")
    
    with st.spinner("🧠 Advisor Agent analyzing..."):
        # NEW PROMPT: Forces the format you requested
        prompt = f"""
        Analyze this portfolio and output your answer strictly in the following format:
        
        ### 🎯 Portfolio Score: {div_score}/100
        
        ### ✅ What is Good
        * [Point 1]
        * [Point 2]
        
        ### ⚠️ What is Missing
        * [Point 1]
        * [Point 2]
        
        ### 🏛️ Final Verdict
        **[BUY / SELL / HOLD]** - [Brief sentence explaining why]
        """
        
        advice_report = st.session_state.team['advisor'].run(advisor_context, prompt)
    
    # Display as clean Markdown
    st.markdown(advice_report)

    # --- TIER 2: AGENCY REPORTS ---
    st.markdown("---")
    st.header("2. Agency Reports")
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("🌍 News Agent (Brave Search)")
        with st.spinner("Searching live web..."):
            ticker_list = ", ".join(list(holdings.keys()))
            st.info(st.session_state.team['news'].run(ticker_list, "Summarize latest market sentiment."))
            
    with c2:
        st.subheader("⚠️ Risk Agent (Quantitative)")
        with st.spinner("Analyzing stats..."):
            risk_context = f"Vol: {volatility.to_dict()}, Corr: {correlation.to_dict()}"
            st.warning(st.session_state.team['risk'].run(risk_context, "Identify key risks."))

    # --- TIER 3: FUTURE SIMULATOR ---
    st.markdown("---")
    st.header("3. Future Simulator (Monte Carlo)")
    st.caption("200 Scenarios for the next 90 days")
    
    sc1, sc2 = st.columns([3, 1])
    with sc1:
        st.line_chart(sim_df.iloc[:, :50]) 
    with sc2:
        final = sim_df.iloc[-1]
        st.metric("Upside (95%)", f"${final.quantile(0.95):,.0f}")
        st.metric("Likely", f"${final.median():,.0f}")
        st.metric("Downside (5%)", f"${final.quantile(0.05):,.0f}")

    # --- TIER 4: OPPORTUNITIES ---
    st.markdown("---")
    st.header("4. AI Opportunities")
    with st.spinner("Recommender Agent searching..."):
        st.success(st.session_state.team['rec'].run(advisor_context))

elif run_btn:
    st.error("Please add stocks.")