import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETERS & SETUP
# ==========================================
np.random.seed(42)
n_paths = 1000          # Number of Monte Carlo simulation paths
n_steps = 10            # Time steps (e.g., 10 years annual)
T = 1.0                 # Time horizon / maturity of a sample swap
r0 = 0.03               # Initial short rate (3%)
sigma = 0.02            # Volatility of interest rates
dt = 1.0                # Time step size (1 year)

# Counterparty parameters
counterparty_pd = 0.01  # Annual probability of default (1%)
recovery_rate = 0.40    # Recovery rate (40%)
discount_rate = 0.03    # Risk-free rate for CVA discounting

# ==========================================
# 2. MONTE CARLO SIMULATION (Vasicek-style paths)
# ==========================================
# Simulate future short rates: r_{t+1} = r_t + sigma * dW
rates = np.zeros((n_paths, n_steps + 1))
rates[:, 0] = r0

for t in range(1, n_steps + 1):
    dW = np.random.normal(0, np.sqrt(dt), n_paths)
    rates[:, t] = rates[:, t-1] + sigma * dW
    # Prevent negative interest rates floor at 0
    rates[:, t] = np.maximum(rates[:, t], 0.0)

# ==========================================
# 3. PORTFOLIO VALUATION & EXPOSURE PROFILE
# ==========================================
# Assume a portfolio whose value depends on interest rates (e.g., a fixed receiver swap)
# Portfolio Value V_t = Notional * (Fixed_Rate - r_t) * Duration_Factor
notional = 10_000_000
fixed_rate = 0.035
duration = 4.0

portfolio_values = np.zeros((n_paths, n_steps + 1))
for t in range(n_steps + 1):
    # If rates drop below fixed rate, the swap has positive mark-to-market (exposure)
    portfolio_values[:, t] = notional * (fixed_rate - rates[:, t]) * duration
    # Exposure is max(0, Value) because you only lose money if the counterparty defaults when you are in-the-money
    portfolio_values[:, t] = np.maximum(portfolio_values[:, t], 0.0)

# Calculate Potential Future Exposure (PFE) at 95th percentile over time
pfe_95 = np.percentile(portfolio_values, 95, axis=0)
expected_exposure = np.mean(portfolio_values, axis=0)

# ==========================================
# 4. CREDIT VALUE ADJUSTMENT (CVA) CALCULATION
# ==========================================
# CVA = (1 - Recovery) * Sum [ EE(t) * Marginal_PD(t) * Discount_Factor(t) ]
cva = 0.0
for t in range(1, n_steps + 1):
    df = np.exp(-discount_rate * t)
    marginal_pd = counterparty_pd  # Simplified constant hazard rate approximation
    ee_t = expected_exposure[t]
    cva += (1.0 - recovery_rate) * ee_t * marginal_pd * df

print(f"--- Monte Carlo PFE & CVA Simulation Complete ---")
print(f"Calculated CVA for Portfolio: ${cva:,.2f}")
print(f"Peak 95% PFE: ${np.max(pfe_95):,.2f}")