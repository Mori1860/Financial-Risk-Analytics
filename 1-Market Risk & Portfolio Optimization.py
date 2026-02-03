import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. SETUP & DATA COLLECTION
# ---------------------------------------------------------
# Selection of Austrian Blue Chips (ATX)
tickers = ['OMV.VI', 'EBS.VI', 'VOE.VI', 'VER.VI', 'ANDR.VI']
print("Downloading ATX market data...")

# auto_adjust=True merges "Adj Close" into the "Close" column automatically
data = yf.download(tickers, start="2020-01-01", end="2025-12-31", auto_adjust=True)
prices = data['Close']

# 2. DATA PRE-PROCESSING
# ---------------------------------------------------------
# Log returns are standard for risk modeling (time-additive)
log_returns = np.log(prices / prices.shift(1)).dropna()

# 3. RISK METRIC: VALUE AT RISK (VaR)
# ---------------------------------------------------------
# Equal weight portfolio assumption for basic VaR
weights_equal = np.array([1/len(tickers)] * len(tickers))      # create an array as a list of same weighted values
portfolio_returns = log_returns.dot(weights_equal)          # .dot() = Matrix multiplication (dot product). 
# Simple explanation: This tells you how your equally-weighted portfolio performed each day based on how each stock moved.

# 95% Parametric VaR
var_95 = np.percentile(portfolio_returns, 5)
print(f"\n--- Risk Metrics ---")
print(f"95% Daily Value at Risk (VaR): {var_95:.2%}")

# 4. MONTE CARLO SIMULATION (Efficient Frontier)
# ---------------------------------------------------------
num_portfolios = 5000
results = []

print(f"Running {num_portfolios} simulations...")
for _ in range(num_portfolios):
    # Random weights
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    
    # Annualized return and volatility (252 trading days)
    p_ret = np.sum(log_returns.mean() * weights) * 252
    p_vol = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))
    
    # Sharpe Ratio (Assuming 0% risk-free rate for simplicity)
    sharpe = p_ret / p_vol
    results.append([p_ret, p_vol, sharpe])

# Convert results list to DataFrame
results_df = pd.DataFrame(results, columns=['Returns', 'Volatility', 'Sharpe_Ratio'])

# Find the Optimal Portfolio (Maximum Sharpe Ratio)
max_sharpe_idx = results_df['Sharpe_Ratio'].idxmax()
optimal_portfolio = results_df.loc[max_sharpe_idx]

# 5. VISUALIZATION
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(results_df.Volatility, results_df.Returns, c=results_df.Sharpe_Ratio, cmap='viridis', alpha=0.5)
plt.colorbar(label='Sharpe Ratio')

# Highlight the Optimal Portfolio with a red star
plt.scatter(optimal_portfolio[1], optimal_portfolio[0], color='red', marker='*', s=250, label='Optimal Portfolio')

plt.title('Efficient Frontier: Austrian Blue-Chips (ATX)')
plt.xlabel('Annualized Volatility (Risk)')
plt.ylabel('Annualized Expected Return')
plt.legend()
plt.grid(True)
plt.show()

print("\n--- Optimization Results ---")
print(f"Optimal Annualized Return: {optimal_portfolio[0]:.2%}")
print(f"Optimal Annualized Volatility: {optimal_portfolio[1]:.2%}")
print(f"Optimal Sharpe Ratio: {optimal_portfolio[2]:.2f}")



