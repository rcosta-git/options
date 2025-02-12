import numpy as np
import scipy.stats as si
import yfinance as yf
from datetime import datetime

"""
* Black-Scholes option pricing model for European call and put options.
* Variables retrieved using yfinance API:
*    - stock price, updates every 1 min
*    - volatility, estimated from 30 day volatility, defaults to 20%
*    - risk-free rate, dynamic using 10-year US treasury yield, defaults to 4.5%
* User inputs include:
*    - ticker
*    - strike price, K
*    - expiration date
"""


"""
* Black-Scholes option pricing model for European call and put options.
* Parameters:
*  - S: Spot price
*  - K: Strike price
*  - T: Time to maturity
*  - r: risk-free interest rate
*  - sigma: volatility of underlying asset
"""
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")

# Get ticker from user
ticker = input("Enter the stock ticker symbol: ").upper()
stock = yf.Ticker(ticker)

# Get real-time stock price
stock_info = stock.history(period="1d", interval="1m")
if stock_info.empty:
    print("Invalid ticker or no data available.")
    exit()

S = stock_info["Close"].iloc[-1]  # Latest closing price
print(f"Current stock price of {ticker}: ${S:.2f}")

# User inputs strike price
K = float(input("Enter the strike price: "))

# User inputs expiration date and calculates time to expiration
expiry_date = input("Enter the option expiration date (YYYY-MM-DD): ")
T = (datetime.strptime(expiry_date, "%Y-%m-%d") - datetime.today()).days / 365
if T <= 0:
    print("Expiration date must be in the future.")
    exit()

# Fetch historical volatility (30-day standard deviation of log returns)
hist = stock.history(period="1mo")["Close"]
if len(hist) > 1:
    log_returns = np.log(hist / hist.shift(1)).dropna()
    sigma = log_returns.std() * np.sqrt(252)  # Annualized volatility
else:
    sigma = 0.2  # Default to 20% if insufficient data
print(f"Estimated Volatility (σ): {sigma:.2%}")

# Get risk-free rate (U.S. 10-Year Treasury Yield)
try:
    treasury = yf.Ticker("^TNX")  # 10-Year Treasury Yield
    r = treasury.history(period="1d")["Close"].iloc[-1] / 100  # Convert percent to decimal
    print(f"Fetched Risk-Free Rate (r): {r:.2%}")
except:
    r = 0.045  # Default to 4.5% if fetch fails
    print("Failed to fetch risk-free rate. Using default 4.5%.")

# Calculate call and put option prices
call_price = black_scholes(S, K, T, r, sigma, "call")
put_price = black_scholes(S, K, T, r, sigma, "put")

# Print results
print(f"\nBlack-Scholes Option Pricing for {ticker}:")
print(f"Call Option Price: ${call_price:.2f}")
print(f"Put Option Price: ${put_price:.2f}")
