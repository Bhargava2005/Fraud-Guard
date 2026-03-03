import pandas as pd
import joblib
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# 1. LOAD MODEL ARTIFACTS
# ═══════════════════════════════════════════════════════════════════
model    = joblib.load("D:\\fraudguard\\server_verson2\\trained_models\\device_risk_models\\device_risk_model.pkl")
scaler   = joblib.load("D:\\fraudguard\\server_verson2\\trained_models\\device_risk_models\\device_risk_scaler.pkl")
iso      = joblib.load("D:\\fraudguard\\server_verson2\\trained_models\\device_risk_models\\device_risk_isolation.pkl")
features = joblib.load("D:\\fraudguard\\server_verson2\\trained_models\\device_risk_models\\device_risk_features.pkl")

# ═══════════════════════════════════════════════════════════════════
# 2. 20 TEST SAMPLES — each row targets a 5% probability band
#
# Scoring logic recap:
#   device_age < 60        → +1.5
#   accounts_per_device>=3 → +2.0
#   emulator_detected==1   → +2.5
#   rooted_or_jailbroken==1→ +2.0
#   vpn_used==1            → +1.5
#   ip_changes_24h>=5      → +1.5
#   geo_distance_km>500    → +1.2
#   login_frequency>30     → +1.0
#   failed_login_attempts>=4→+1.5
#   threshold = 6, max = 14.7
#
# Columns:
#   device_age_days, device_type, accounts_per_device,
#   emulator_detected, rooted_or_jailbroken, vpn_used,
#   ip_changes_24h, geo_distance_km, login_frequency,
#   failed_login_attempts
#
# Target bands:
#   Row 1  →  1– 5%   Row 2  →  5–10%   Row 3  → 10–15%   Row 4  → 15–20%
#   Row 5  → 20–25%   Row 6  → 25–30%   Row 7  → 30–35%   Row 8  → 35–40%
#   Row 9  → 40–45%   Row 10 → 45–50%   Row 11 → 50–55%   Row 12 → 55–60%
#   Row 13 → 60–65%   Row 14 → 65–70%   Row 15 → 70–75%   Row 16 → 75–80%
#   Row 17 → 80–85%   Row 18 → 85–90%   Row 19 → 90–95%   Row 20 → 95–100%
# ═══════════════════════════════════════════════════════════════════

test_cases_20x10 = [
# [dev_age, dev_type, acc/dev, emulator, rooted, vpn, ip_chg, geo_km, login_freq, fail_login]
  [1400,    2,        0,       1,        1,      0,   1,      200,    10,         0 ],  # R1   1–5%    score≈0.0
  [800,     4,        0,       1,        0,      0,   2,      550,    12,         6 ],  # R2   5–10%   score≈0.0
  [500,     0,        1,       0,        0,      0,   3,      400,    15,         1 ],  # R3  10–15%   score≈0.0
  [55,      1,        0,       0,        0,      0,   2,      300,    14,         1 ],  # R4  15–20%   score≈1.5 (new device)
  [50,      2,        1,       0,        0,      0,   2,      350,    18,         2 ],  # R5  20–25%   score≈1.5
  [45,      0,        1,       0,        0,      1,   2,      300,    16,         1 ],  # R6  25–30%   score≈3.0 (new+vpn)
  [40,      1,        1,       0,        0,      1,   3,      400,    20,         2 ],  # R7  30–35%   score≈3.0
  [35,      0,        2,       0,        0,      1,   3,      450,    22,         2 ],  # R8  35–40%   score≈3.0
  [30,      2,        2,       0,        0,      1,   4,      400,    25,         3 ],  # R9  40–45%   score≈3.0
  [25,      0,        2,       0,        0,      1,   4,      600,    28,         3 ],  # R10 45–50%   score≈4.2 (new+vpn+geo)
  [900,     1,        3,       0,        0,      1,   4,      600,    28,         3 ],  # R11 50–55%   score≈4.7 (acc+vpn+geo)
  [20,      0,        2,       0,        0,      1,   5,      700,    28,         3 ],  # R12 55–60%   score≈5.2 (new+vpn+ip+geo)
  [15,      1,        3,       0,        0,      1,   5,      800,    32,         3 ],  # R13 60–65%   score≈7.2 (new+acc+vpn+ip+geo+freq)
  [12,      0,        3,       0,        0,      1,   5,      900,    35,         3 ],  # R14 65–70%   score≈7.2
  [10,      2,        3,       0,        1,      1,   5,      1000,   33,         3 ],  # R15 70–75%   score≈9.2 (new+acc+rooted+vpn+ip+geo+freq)
  [80,       1,        2,       0,        1,      1,   3,     400,   40,         4 ],  # R16 75–80%   score≈10.7 (new+acc+rooted+vpn+ip+geo+freq+fail)
  [6,       1,        3,       1,        0,      1,   6,      1400,   36,         4 ],  # R17 80–85%   score≈11.5 (new+acc+emulator+vpn+ip+geo+freq+fail)
  [500,       0,        1,       0,        1,      0,   7,      1800,   38,         5 ],  # R18 85–90%   score≈13.2
  [4,       0,        4,       0,        0,      1,   8,      250,   40,         6 ],  # R19 90–95%   score≈13.7
  [900,       2,        0,       1,        1,      0,   3,      280,   45,         4 ],  # R20 95–100%  score≈14.7 (all flags)
]

TARGET_BANDS = [
    "1–5%",   "5–10%",  "10–15%", "15–20%", "20–25%",
    "25–30%", "30–35%", "35–40%", "40–45%", "45–50%",
    "50–55%", "55–60%", "60–65%", "65–70%", "70–75%",
    "75–80%", "80–85%", "85–90%", "90–95%", "95–100%",
]

# ═══════════════════════════════════════════════════════════════════
# 3. PREPARE INPUT
# ═══════════════════════════════════════════════════════════════════
# Use only the original 10 features for IsolationForest
original_features = [
    "device_age_days", "device_type", "accounts_per_device",
    "emulator_detected", "rooted_or_jailbroken", "vpn_used",
    "ip_changes_24h", "geo_distance_km", "login_frequency",
    "failed_login_attempts"
]

input_df = pd.DataFrame(test_cases_20x10, columns=original_features)

# Add anomaly score (same as training pipeline)
input_df["anomaly_score"] = iso.predict(input_df[original_features])

# Align column order exactly as training
input_df = input_df[features]

# Scale
input_scaled = scaler.transform(input_df)

# ═══════════════════════════════════════════════════════════════════
# 4. PREDICT
# ═══════════════════════════════════════════════════════════════════
predicted_labels        = model.predict(input_scaled)
predicted_probabilities = model.predict_proba(input_scaled)[:, 1]

# ═══════════════════════════════════════════════════════════════════
# 5. DISPLAY RESULTS
# ═══════════════════════════════════════════════════════════════════
print("=" * 80)
print("         🔍 DEVICE RISK PREDICTION — 20 TEST SAMPLES")
print("=" * 80)
print(f"  {'#':<4} {'Target Band':<13} {'Risk %':<10} {'Visual Bar':<25} {'Label'}")
print("-" * 80)

for i, (prob, label, band) in enumerate(zip(predicted_probabilities, predicted_labels, TARGET_BANDS)):
    status = "🔴 HIGH RISK (1)" if label == 1 else "🟢 LOW RISK  (0)"
    bar    = "█" * int(prob * 25)
    print(f"  {i+1:<4} {band:<13} {prob*100:>6.2f}%   {bar:<25} {status}")

print("=" * 80)

# Summary stats
print(f"\n  📊 Probability Range  : {predicted_probabilities.min()*100:.2f}% → {predicted_probabilities.max()*100:.2f}%")
print(f"  📊 High Risk (1) count: {predicted_labels.sum()} / 20")
print(f"  📊 Low Risk  (0) count: {(predicted_labels==0).sum()} / 20")
print("=" * 80)