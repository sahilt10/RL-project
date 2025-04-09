import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_price_data
from q_learning import simulate_agent

st.set_page_config(layout="wide")
st.title("📈 Multi-Stock Portfolio Optimizer (Q-learning)")

tickers = st.text_input("Enter stock tickers separated by comma (e.g., AAPL, MSFT, GOOGL):", "AAPL,MSFT")
start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-01-01"))

if st.button("Optimize Portfolio"):
    data = load_price_data(tickers, start_date, end_date)

    if data is None:
        st.error("❌ Failed to load data.")
    else:
        st.success("✅ Data loaded. Simulating agent...")
        portfolio_values, transactions = simulate_agent(data)

        # Buy & hold baseline
        initial_prices = data.iloc[0]
        final_prices = data.iloc[len(portfolio_values) - 1]
        shares_bought = 1000 / initial_prices
        final_value_hold = (shares_bought * final_prices).sum()

        start_value = 1000
        final_value_rl = portfolio_values[-1]

        st.subheader("💼 Portfolio Performance Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("🚀 Starting Portfolio Value", f"${start_value:.2f}")
        col2.metric("🎯 Optimized Final Value (RL)", f"${final_value_rl:.2f}")
        col3.metric("📊 Buy & Hold Value", f"${final_value_hold:.2f}")

        # Plot portfolio value
        fig_val, ax_val = plt.subplots(figsize=(10, 4))
        ax_val.plot(portfolio_values, label="RL Portfolio", color="green")
        ax_val.set_title("Portfolio Value Over Time")
        ax_val.set_ylabel("Value ($)")
        ax_val.set_xlabel("Time Step")
        ax_val.legend()
        st.pyplot(fig_val)

        # Show transaction log
        st.subheader("🧾 Transaction Log")
        if transactions:
            df_log = pd.DataFrame(transactions, columns=["Step", "Stock", "Action", "Price"])
            st.dataframe(df_log)
        else:
            st.write("No transactions made.")

        # Price history plot
        st.subheader("📈 Stock Price History")
        fig_prices, ax_prices = plt.subplots(figsize=(10, 4))
        data.plot(ax=ax_prices)
        ax_prices.set_ylabel("Price ($)")
        st.pyplot(fig_prices)
