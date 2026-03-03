import pandas as pd
import joblib
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# 1. LOAD MODEL ARTIFACTS
# ═══════════════════════════════════════════════════════════════════
model    = joblib.load("D:\\fraudguard\\server_verson2\\trained_models\\seller_risk_models\\seller_risk_model.pkl")
scaler   = joblib.load("D:\\fraudguard\\server_verson2\\trained_models\\seller_risk_models\\seller_risk_scaler.pkl")
features = joblib.load("D:\\fraudguard\\server_verson2\\trained_models\\seller_risk_models\\seller_risk_features.pkl")

# ═══════════════════════════════════════════════════════════════════
# 2. 20 TEST SAMPLES — each row targets a strict 5% probability band
#
# Scoring logic recap:
#   seller_age < 180              → +1.5
#   verification_level == 0       → +2.0
#   seller_return_rate > 0.35     → +2.0
#   seller_dispute_rate > 0.20    → +1.5
#   wrong_item_rate > 0.25        → +1.2
#   damaged_item_rate > 0.25      → +1.2
#   refund_before_inspection == 1 → +1.0
#   negative_feedback_rate > 0.30 → +2.0
#   threshold = 5,  max = 12.4
#
# Columns:
#   seller_age_days, verification_level, total_orders,
#   seller_return_rate, seller_dispute_rate, wrong_item_rate,
#   damaged_item_rate, avg_product_price,
#   refund_before_inspection, negative_feedback_rate
# ═══════════════════════════════════════════════════════════════════

#   [ age,   verif, orders, ret_r, disp,  wrong, damaged, avg_price, refund, neg_fb ]
samples = [
    [900,     2,   2000,  0.15,  0.12,  0.14,   0.14,  4000.00,   0,    0.15 ],  # R3  10–15%   score=0.0
    [150,     2,   1000,  0.18,  0.14,  0.16,   0.16,  3000.00,   1,    0.18 ],  # R6  25–30%   score=2.5  (+age+refund)
    [150,     2,    800,  0.18,  0.22,  0.28,   0.16,  2600.00,   0,    0.18 ],  # R8  35–40%   score=4.2  (+age+disp+wrong)
    [150,     2,    700,  0.18,  0.22,  0.28,   0.28,  2400.00,   0,    0.18 ],  # R9  40–45%   score=5.4  (+damaged)
    [150,     2,    600,  0.18,  0.22,  0.28,   0.28,  2200.00,   1,    0.18 ],  # R10 45–50%   score=6.4  (+refund)
    [190,     1,    500,  0.18,  0.22,  0.28,   0.28,  2000.00,   1,    0.32 ],  # R11 50–55%   score=6.4+2.0 (+neg_fb)
    [200,     1,    400,  0.34,  0.22,  0.28,   0.28,  1600.00,   1,    0.22 ],  # R13 60–65%   score=+2.0 (+return)
    [230,     1,    350,  0.25,  0.15,  0.30,   0.28,  1400.00,   1,    0.25 ],  # R14 65–70%
    [300,     1,    300,  0.31,  0.28,  0.30,   0.50,  1200.00,   1,    0.28 ],  # R15 70–75%
    [40,      0,    150,  0.60,  0.35,  0.36,   0.36,   600.00,   1,    0.35 ],  # R18 85–90%   score=12.4 (+neg_fb>0.30)
]

TARGET_BANDS = [
    "0–5%",   "5–10%",  "10–15%", "15–20%", "20–25%",
    "25–30%", "30–35%", "35–40%", "40–45%", "45–50%",
    "50–55%", "55–60%", "60–65%", "65–70%", "70–75%",
    "75–80%", "80–85%", "85–90%", "90–95%", "95–100%",
]

# ═══════════════════════════════════════════════════════════════════
# 3. PREPARE, SCALE & PREDICT  (simple pipeline — no mixed types)
# ═══════════════════════════════════════════════════════════════════
input_df     = pd.DataFrame(samples, columns=features)
input_scaled = scaler.transform(input_df)

predicted_labels        = model.predict(input_scaled)
predicted_probabilities = model.predict_proba(input_scaled)[:, 1]

# ═══════════════════════════════════════════════════════════════════
# 4. DISPLAY RESULTS
# ═══════════════════════════════════════════════════════════════════
print("=" * 82)
print("        🏪 SELLER RISK PREDICTION — 20 TEST SAMPLES")
print("=" * 82)
print(f"  {'#':<4} {'Target Band':<13} {'Risk %':<10} {'Visual Bar':<26} {'Label'}")
print("-" * 82)
for i,(prob,label,band) in enumerate(zip(predicted_probabilities,predicted_labels,TARGET_BANDS)):
    status = "🔴 HIGH RISK (1)" if label == 1 else "🟢 LOW RISK  (0)"
    bar    = "█" * int(prob * 25)
    print(f"  {i+1:<4} {band:<13} {prob*100:>6.2f}%   {bar:<26} {status}")
print("=" * 82)

print(f"\n  📊 Probability Range   : {predicted_probabilities.min()*100:.2f}% → {predicted_probabilities.max()*100:.2f}%")
print(f"  📊 High Risk (1) count : {predicted_labels.sum()} / 20")
print(f"  📊 Low  Risk (0) count : {(predicted_labels==0).sum()} / 20")
spread_ok = predicted_probabilities.max() > 0.90 and predicted_probabilities.min() < 0.10
print(f"  📊 Full 0–100% Spread  : {'✅ YES' if spread_ok else '⚠️  Partial'}")
print("\n  verification_level: 0=Unverified  1=Partially Verified  2=Fully Verified")
print("  Seller risk threshold = 5  (more sensitive than other models)")