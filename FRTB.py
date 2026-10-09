import numpy as np
import pandas as pd

# ==========================================
# 1. DEFINE PORTFOLIO SENSITIVITIES & BUCKETS
# ==========================================
# Example: Interest Rate Risk (Delta Sensitivities across tenor buckets)
# Buckets: [0.5yr, 1yr, 3yr, 5yr, 10yr]
buckets = ['0.5Y', '1Y', '3Y', '5Y', '10Y']
delta_sensitivities = np.array([100_000, 250_000, -150_000, 400_000, -200_000]) # in currency units

# Regulatory Risk Weights (RW) for General Interest Rate Risk (GIRR) under FRTB-SA
risk_weights = np.array([0.015, 0.015, 0.02, 0.025, 0.03]) # 1.5% to 3%

# ==========================================
# 2. APPLY RISK WEIGHTS TO CALCULATE WEIGHTED SENSITIVITIES
# ==========================================
weighted_sensitivities = delta_sensitivities * risk_weights

# ==========================================
# 3. INTRA-BUCKET CORRELATION AGGREGATION
# ==========================================
# FRTB correlation matrix parameter (simplified uniform correlation rho = 50% within bucket/across tenors)
rho = 0.5
n = len(buckets)

# Build correlation matrix C where C_ij = rho (except diagonal which is 1.0)
corr_matrix = np.full((n, n), rho)
np.fill_diagonal(corr_matrix, 1.0)

# Capital formula within a risk class: K_b = sqrt( sum_i (WS_i^2) + sum_i sum_{j != i} (WS_i * WS_j * rho_ij) )
variance_term = np.dot(weighted_sensitivities, np.dot(corr_matrix, weighted_sensitivities))
capital_charge_risk_class = np.sqrt(np.maximum(variance_term, 0.0))

print(f"--- FRTB Standardized Approach (SA) Calculation ---")
print(f"Weighted Sensitivities: {weighted_sensitivities}")
print(f"Aggregated Capital Charge for Interest Rate Risk Class: ${capital_charge_risk_class:,.2f}")