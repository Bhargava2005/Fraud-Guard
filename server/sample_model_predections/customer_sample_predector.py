import pandas as pd
import joblib
import numpy as np

# 1. Load saved model, scaler, and feature list
model    = joblib.load("D:\\fraudguard\\server_verson2\\trained_models\\customer_risk_models\\customer_risk_model.pkl")
scaler   = joblib.load("D:\\fraudguard\\server_verson2\\trained_models\\customer_risk_models\\customer_risk_scaler.pkl")
features = joblib.load("D:\\fraudguard\\server_verson2\\trained_models\\customer_risk_models\\customer_risk_features.pkl")

# 2. 10x10 2D Array — each row is one test case
# Columns (10 features):
#   account_age_days, total_orders, return_rate, avg_order_value,
#   cod_percentage, end_window_returns, damaged_claim_rate,
#   linked_accounts, vpn_usage, past_fraud_flag
#
# Risk score contribution per flag:
#   account_age < 90   → +1.5
#   return_rate > 0.4  → +2.0
#   cod_pct > 0.6      → +1.0
#   end_window > 0.5   → +1.2
#   damaged > 0.5      → +1.3
#   linked >= 3        → +2.0
#   vpn == 1           → +1.0
#   past_fraud == 1    → +3.0
#
# Row-wise score targets:
#   Row 1  (0–10%)   score ≈ 0.0  — everything clean
#   Row 2  (11–20%)  score ≈ 1.5  — only new account
#   Row 3  (21–30%)  score ≈ 2.5  — new account + vpn
#   Row 4  (31–40%)  score ≈ 3.5  — new account + vpn + cod
#   Row 5  (41–50%)  score ≈ 4.5  — new account + return + vpn
#   Row 6  (51–60%)  score ≈ 5.5  — return + cod + end_window + damaged
#   Row 7  (61–70%)  score ≈ 6.3  — return + linked + vpn + end_window
#   Row 8  (71–80%)  score ≈ 7.5  — return + linked + vpn + damaged + new account
#   Row 9  (81–90%)  score ≈ 9.0  — past_fraud + return + linked + vpn + new account
#   Row 10 (91–100%) score ≈ 11.5 — all flags triggered

test_cases_2d = [
#    acct_age  orders  ret_rate  avg_val   cod_pct  end_win  damaged  linked  vpn  fraud
    [1200,     80,     0.10,     3500.00,  0.20,    0.15,    0.10,    0,      0,   0  ],  # Row 1  → 0–10%
    [75,       12,     0.20,     1800.00,  0.30,    0.20,    0.20,    0,      0,   0  ],  # Row 2  → 11–20%
    [70,       10,     0.22,     1500.00,  0.35,    0.25,    0.22,    0,      1,   0  ],  # Row 3  → 21–30%
    [65,       8,      0.25,     1200.00,  0.65,    0.30,    0.25,    0,      1,   0  ],  # Row 4  → 31–40%
    [80,       9,      0.42,     2000.00,  0.55,    0.45,    0.30,    1,      1,   0  ],  # Row 5  → 41–50%
    [500,      40,     0.50,     2500.00,  0.65,    0.55,    0.55,    1,      0,   0  ],  # Row 6  → 51–60%
    [200,      25,     0.55,     1800.00,  0.58,    0.60,    0.45,    3,      1,   0  ],  # Row 7  → 61–70%
    [60,       7,      0.65,     900.00,   0.70,    0.62,    0.58,    3,      1,   0  ],  # Row 8  → 71–80%
    [55,       5,      0.72,     700.00,   0.75,    0.68,    0.62,    3,      1,   1  ],  # Row 9  → 81–90%
    [30,       3,      0.90,     400.00,   0.95,    0.88,    0.85,    5,      1,   1  ],  # Row 10 → 91–100%
]

# 3. Convert to DataFrame with correct feature order
input_df = pd.DataFrame(test_cases_2d, columns=features)

# 4. Scale using the same scaler from training
input_scaled = scaler.transform(input_df)

# 5. Predict all 10 rows at once
predicted_labels        = model.predict(input_scaled)
predicted_probabilities = model.predict_proba(input_scaled)[:, 1]

# 6. Display Results Table
bands = [
    "0–10%",   "11–20%",  "21–30%",  "31–40%",  "41–50%",
    "51–60%",  "61–70%",  "71–80%",  "81–90%",  "91–100%"
]

print("=" * 72)
print("         🔍 CUSTOMER RISK PREDICTION — 10×10 TEST CASES")
print("=" * 72)
print(f"  {'Row':<4} {'Target Band':<13} {'Risk %':<10} {'Bar':<22} {'Label'}")
print("-" * 72)

for i, (prob, label, band) in enumerate(zip(predicted_probabilities, predicted_labels, bands)):
    status = "🔴 HIGH RISK (1)" if label == 1 else "🟢 LOW RISK  (0)"
    bar    = "█" * int(prob * 20)
    print(f"  R{i+1:<3} {band:<13} {prob*100:>6.2f}%   {bar:<22} {status}")

print("=" * 72)

# 7. Print the raw 2D array
print("\n📦 Raw 10×10 Input Array:")
header = ["acct_age", "orders", "ret_rate", "avg_val", "cod_pct",
          "end_win", "damaged", "linked", "vpn", "fraud"]
print(f"  {'':>4}", end="")
for h in header:
    print(f"  {h:>9}", end="")
print()
print("  " + "-" * 105)
for i, row in enumerate(test_cases_2d):
    print(f"  R{i+1:<3}", end="")
    for val in row:
        print(f"  {str(val):>9}", end="")
    print()