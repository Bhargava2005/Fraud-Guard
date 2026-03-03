import pandas as pd
import joblib
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# 1. LOAD MODEL ARTIFACTS
# ═══════════════════════════════════════════════════════════════════
model    = joblib.load("D:\\fraudguard\\server_verson2\\trained_models\\logistic_risk_models\\logistics_risk_model.pkl")
scaler   = joblib.load("D:\\fraudguard\\server_verson2\\trained_models\\logistic_risk_models\\logistics_risk_scaler.pkl")
features = joblib.load("D:\\fraudguard\\server_verson2\\trained_models\\logistic_risk_models\\logistics_risk_features.pkl")

# ═══════════════════════════════════════════════════════════════════
# 2. 20 TEST SAMPLES — each targets a 5% probability band
#
# Scoring logic recap:
#   courier_risk > 0.6       → +2.0
#   delivery_attempts >= 3   → +1.5
#   delivery_delay > 5       → +1.2
#   otp_confirmation == 0    → +1.5
#   delivery_photo == 0      → +1.2
#   pickup_delay > 5         → +1.0
#   pickup_attempts >= 3     → +1.2
#   tamper_route == 1        → +1.5
#   distance_km > 800        → +1.0
#   weight_mismatch == 1     → +2.0
#   threshold = 6,  max = 14.1
#
# Columns:
#   courier_risk_score, delivery_attempts, delivery_delay_days,
#   otp_confirmation, delivery_photo, pickup_delay_days,
#   pickup_attempts, tamper_route, distance_km, weight_mismatch
#
# Target bands per row:
#   R1  →  0–5%    R2  →  5–10%   R3  → 10–15%   R4  → 15–20%
#   R5  → 20–25%   R6  → 25–30%   R7  → 30–35%   R8  → 35–40%
#   R9  → 40–45%   R10 → 45–50%   R11 → 50–55%   R12 → 55–60%
#   R13 → 60–65%   R14 → 65–70%   R15 → 70–75%   R16 → 75–80%
#   R17 → 80–85%   R18 → 85–90%   R19 → 90–95%   R20 → 95–100%
# ═══════════════════════════════════════════════════════════════════

#   [cour_risk, del_att, del_dly, otp, photo, pkp_dly, pkp_att, tamper, dist_km, wt_mis]
samples = [
    [0.50,     2,       4,       1,    1,     4,       2,       0,      600,     0],  # R4  15–20%   score≈0.0
    [0.70,     2,       6,       1,    1,     4,       2,       0,      700,     0],  # R6  25–30%   score≈3.2 (courier+delay)
    [0.70,     2,       6,       1,    0,     4,       2,       0,      750,     0],  # R7  30–35%   score≈4.4 (courier+delay+photo)
    [0.52,     3,       6,       1,    0,     4,       2,       0,      800,     0],  # R8  35–40%   score≈5.9 (courier+att+delay+photo)
    [0.72,     3,       6,       1,    0,     5,       2,       0,      850,     0],  # R9  40–45%   score≈5.9
    [0.72,     3,       6,       1,    0,     6,       2,       0,      900,     0],  # R10 45–50%   score≈6.9 (courier+att+delay+photo+pkp+dist)
    [0.75,     3,       6,       0,    0,     6,       2,       0,      900,     0],  # R11 50–55%   score≈8.4 (above+otp)
    [0.78,     3,       7,       0,    0,     7,       3,       0,     1000,     0],  # R13 60–65%   score≈9.6
    [0.93,     4,      12,       1,    0,    11,       2,       0,     750,     0],  # R19 90–95%   score≈13.1
    [0.48,     2,      4,       1,    1,    4,       4,       1,     2400,     1],  # R20 95–100%  score≈14.1 all flags
]

TARGET_BANDS = [
    "0–5%",   "5–10%",  "10–15%", "15–20%", "20–25%",
    "25–30%", "30–35%", "35–40%", "40–45%", "45–50%",
    "50–55%", "55–60%", "60–65%", "65–70%", "70–75%",
    "75–80%", "80–85%", "85–90%", "90–95%", "95–100%",
]

# ═══════════════════════════════════════════════════════════════════
# 3. PREPARE, SCALE & PREDICT
# ═══════════════════════════════════════════════════════════════════
input_df     = pd.DataFrame(samples, columns=features)
input_scaled = scaler.transform(input_df)

predicted_labels        = model.predict(input_scaled)
predicted_probabilities = model.predict_proba(input_scaled)[:, 1]

# ═══════════════════════════════════════════════════════════════════
# 4. DISPLAY RESULTS
# ═══════════════════════════════════════════════════════════════════
print("=" * 82)
print("        🚚 LOGISTICS RISK PREDICTION — 20 TEST SAMPLES")
print("=" * 82)
print(f"  {'#':<4} {'Target Band':<13} {'Risk %':<10} {'Visual Bar':<26} {'Label'}")
print("-" * 82)

for i, (prob, label, band) in enumerate(zip(predicted_probabilities, predicted_labels, TARGET_BANDS)):
    status = "🔴 HIGH RISK (1)" if label == 1 else "🟢 LOW RISK  (0)"
    bar    = "█" * int(prob * 25)
    print(f"  {i+1:<4} {band:<13} {prob*100:>6.2f}%   {bar:<26} {status}")

print("=" * 82)
print(f"\n  📊 Probability Range   : {predicted_probabilities.min()*100:.2f}% → {predicted_probabilities.max()*100:.2f}%")
print(f"  📊 High Risk (1) count : {predicted_labels.sum()} / 20")
print(f"  📊 Low  Risk (0) count : {(predicted_labels==0).sum()} / 20")
print("=" * 82)

# Key flag summary per row for reference
print("\n  📋 KEY FLAGS ACTIVE PER ROW:")
print(f"  {'Row':<5} {'courier':>8} {'del_att':>8} {'del_dly':>8} {'otp':>5} {'photo':>6} {'tamper':>7} {'wt_mis':>7} {'raw_score':>10}")
print("  " + "-"*68)
flag_names = ["courier_risk_score","delivery_attempts","delivery_delay_days",
              "otp_confirmation","delivery_photo","tamper_route","weight_mismatch"]
for i, row in enumerate(samples):
    cr,da,dd,otp,ph,pkd,pka,tr,dk,wm = row
    score = (2.0 if cr>0.6 else 0)+(1.5 if da>=3 else 0)+(1.2 if dd>5 else 0)+(1.5 if otp==0 else 0)+(1.2 if ph==0 else 0)+(1.0 if pkd>5 else 0)+(1.2 if pka>=3 else 0)+(1.5 if tr==1 else 0)+(1.0 if dk>800 else 0)+(2.0 if wm==1 else 0)
    print(f"  R{i+1:<4} {cr:>8.2f} {da:>8} {dd:>8} {otp:>5} {ph:>6} {tr:>7} {wm:>7} {score:>10.1f}")
