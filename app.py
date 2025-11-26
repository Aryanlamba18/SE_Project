import streamlit as st
import pandas as pd
import plotly.express as px
from data_engine import fetch_market_data, calculate_portfolio_metrics, run_monte_carlo
from ai_brain import get_agent_response, get_stock_recommendations

st.set_page_config(page_title="IntelliQuant Suite", layout="wide")

st.title("🤖 IntelliQuant: Agentic Finance Suite")
st.markdown("### Team 2 - Feasibility Prototype")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Your Portfolio")
    
    default_data = pd.DataFrame([
        {"Ticker": "AAPL", "Quantity": 10, "Avg Buy Price": 150.0},
        {"Ticker": "MSFT", "Quantity": 5, "Avg Buy Price": 300.0},
        {"Ticker": "GOOGL", "Quantity": 8, "Avg Buy Price": 100.0}
    ])

    edited_df = st.data_editor(
        default_data, 
        num_rows="dynamic", 
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker Symbol", required=True),
            "Quantity": st.column_config.NumberColumn("Shares", min_value=1, required=True),
            "Avg Buy Price": st.column_config.NumberColumn("Avg Buy Price ($)", min_value=0.1, required=True)
        },
        key="portfolio_editor"
    )

    holdings = {}
    for index, row in edited_df.iterrows():
        if row["Ticker"] and row["Quantity"] > 0:
            holdings[row["Ticker"].strip().upper()] = {
                "qty": row["Quantity"],
                "buy_price": row["Avg Buy Price"]
            }
    
    run_btn = st.button("🚀 Run Analysis")

# --- MAIN LOGIC ---
if run_btn and len(holdings) > 0:
    # 1. Fetch Data
    with st.spinner("🤖 Data Agent Fetching Market Data..."):
        history, benchmark, fundamentals, news = fetch_market_data(holdings)
        
    # 2. Analytics
    with st.spinner("📊 Risk Agent Calculating Metrics..."):
        volatility, correlation, div_score, comparison_df = calculate_portfolio_metrics(history, benchmark)

    # 3. Run Monte Carlo Simulation (New Step)
    with st.spinner("🔮 Generating Future Scenarios..."):
        # Calculate weights based on current position value
        weights = dict(zip(fundamentals['Ticker'], fundamentals['Position Value']))
        total_value = fundamentals['Position Value'].sum()
        
        sim_days = 90 # Simulate next 3 months
        sim_df = run_monte_carlo(history, weights, total_value, days=sim_days, simulations=200)

    # Context for Advisor
    total_cost = fundamentals['Cost Basis'].sum()
    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
    
    # Calculate Simulation Stats
    final_values = sim_df.iloc[-1]
    worst_case = final_values.quantile(0.05) # 5th percentile
    best_case = final_values.quantile(0.95)  # 95th percentile
    median_case = final_values.median()

    advisor_context = f"""
    Portfolio Value: ${total_value:,.2f}
    P&L: ${total_pnl:,.2f} ({total_pnl_pct:.2f}%)
    Diversification Score: {div_score}/100
    Holdings: {fundamentals[['Ticker', 'Position Value']].to_dict()}
    
    Future Projection (90 Days):
    - Worst Case (95% confidence): ${worst_case:,.2f}
    - Best Case (Upside): ${best_case:,.2f}
    """

    # --- TIER 1: OVERVIEW ---
    st.header("1. Portfolio Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Net Worth", f"${total_value:,.0f}")
    m2.metric("Total P&L", f"${total_pnl:,.0f}", f"{total_pnl_pct:.1f}%")
    m3.metric("Diversification", f"{div_score}/100")
    m4.metric("Avg Volatility", f"{volatility.mean():.2f}")
    
    with st.spinner("🧠 Advisor Agent Thinking..."):
        advice = get_agent_response("Advisor Agent", advisor_context, "Give a single word recommendation (BUY/SELL/HOLD) followed by a short reason.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if "BUY" in advice.upper():
            st.success(f"## ✅ STRONG BUY SIGNAL")
        elif "SELL" in advice.upper():
            st.error(f"## 🔴 SELL SIGNAL")
        else:
            st.warning(f"## ⚠️ HOLD / CAUTION")
        st.write(f"**Verdict:** {advice}")

    # --- TIER 2: FUTURE SIMULATION (New Feature) ---
    st.markdown("---")
    st.header(f"🔮 Future Simulator (Next {sim_days} Days)")
    st.caption("Monte Carlo Simulation: 200 possible market scenarios based on your portfolio's historical volatility.")
    
    sc1, sc2 = st.columns([3, 1])
    with sc1:
        # Plot only the first 50 lines to keep it clean, but calculate stats on all 200
        st.line_chart(sim_df.iloc[:, :50], use_container_width=True)
    
    with sc2:
        st.subheader("Projections")
        st.metric("🚀 Best Case", f"${best_case:,.0f}", delta=f"+{((best_case/total_value)-1)*100:.1f}%")
        st.metric("⚖️ Likely Outcome", f"${median_case:,.0f}", delta=f"+{((median_case/total_value)-1)*100:.1f}%")
        st.metric("⚠️ Worst Case", f"${worst_case:,.0f}", delta=f"{((worst_case/total_value)-1)*100:.1f}%", delta_color="inverse")
        st.info("The 'Worst Case' indicates the Value at Risk (VaR) if the market turns against you significantly.")

    # --- TIER 3: BENCHMARK & RECOMMENDER ---
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📈 vs S&P 500 (Past Year)")
        st.line_chart(comparison_df, color=["#00FF00", "#FF0000"])
    
    with c2:
        st.subheader("💡 AI Opportunities")
        with st.spinner("Scanning..."):
            st.info(get_stock_recommendations(advisor_context))

    # --- TIER 4: DEEP DIVE ---
    st.markdown("---")
    st.header("2. Deep Dive Analysis")
    tab1, tab2 = st.tabs(["📊 Visuals & Data", "🧠 Agent Reports"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.pie(fundamentals, names='Ticker', values='Position Value', hole=0.4, title="Allocation"), use_container_width=True)
        with c2:
            colors = ['green' if x > 0 else 'red' for x in fundamentals['Unrealized P&L']]
            st.plotly_chart(px.bar(fundamentals, x='Ticker', y='Unrealized P&L', color_discrete_sequence=colors, title="P&L Breakdown"), use_container_width=True)
        with st.expander("View Detailed Fundamentals"):
            st.dataframe(fundamentals)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("News Agent")
            with st.spinner("Reading news..."):
                st.write(get_agent_response("News Agent", str(news), "Summarize sentiment."))
        with col2:
            st.subheader("Risk Agent")
            with st.spinner("Analyzing risk..."):
                st.write(get_agent_response("Risk Agent", f"Correlations: {correlation.to_dict()}", "Analyze risks."))

elif run_btn:
    st.error("Please add at least one stock to your portfolio.")