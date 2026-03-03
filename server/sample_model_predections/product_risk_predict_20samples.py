import pandas as pd
import joblib
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# 1. LOAD MODEL ARTIFACTS
# ═══════════════════════════════════════════════════════════════════
model    = joblib.load("D:\\fraudguard\\server_verson2\\trained_models\\product_risk_models\\product_risk_model.pkl")
scaler   = joblib.load("D:\\fraudguard\\server_verson2\\trained_models\\product_risk_models\\product_risk_scaler.pkl")
features = joblib.load("D:\\fraudguard\\server_verson2\\trained_models\\product_risk_models\\product_risk_features.pkl")

# ═══════════════════════════════════════════════════════════════════
# 2. 20 TEST SAMPLES — each row targets a strict 5% probability band
#
# Scoring logic recap:
#   product_price > 15000        → +2.0
#   product_category == 2        → +1.5
#   return_rate > 0.35           → +2.0
#   fraud_return_rate > 0.20     → +2.0
#   avg_days_to_return > 20      → +1.0
#   serial_tracked == 0          → +1.5
#   counterfeit_risk > 0.5       → +1.5
#   fragile == 1                 → +1.0
#   seller_product_risk > 0.6    → +1.5
#   discount_percentage > 0.5    → +1.0
#   threshold = 6,  max = 15.0
#
# Design strategy:
#   Rows 1–4   : score = 0.0      → 0–20%   (all flags off)
#   Rows 5–6   : score = 1.0–1.5  → 20–30%  (add weak flags)
#   Rows 7–8   : score = 2.5–3.0  → 30–40%  (add 2nd flag)
#   Rows 9–10  : score = 3.5–4.5  → 40–50%  (approach threshold)
#   Rows 11–12 : score = 6.0–7.5  → 50–60%  (cross threshold)
#   Rows 13–14 : score = 8.5–9.5  → 60–70%  (multiple triggers)
#   Rows 15–16 : score = 10.5–11.5 → 70–80% (many flags)
#   Rows 17–18 : score = 12.5–13.5 → 80–90% (most flags)
#   Rows 19–20 : score = 14.0–15.0 → 90–100% (all flags)
#
# Columns (must match training feature order):
#   product_price, product_category, return_rate, fraud_return_rate,
#   avg_days_to_return, serial_tracked, counterfeit_risk, fragile,
#   seller_product_risk, discount_percentage
# ═══════════════════════════════════════════════════════════════════

#   [ price,    cat, ret_r, fraud_r, days, serial, ctft,  frag, sell_r, disc  ]
samples = [

    [5000.00,    0,   0.15,   0.08,    8,     1,    0.25,    0,    0.28,   0.20 ],  # R3  10–15%   score=0.0
  
    # ── LOW-MEDIUM: 2 flags ─────────────────────────────────────────────────────────
 
    [8000.00,    1,   0.20,   0.10,   22,     0,    0.30,    1,    0.30,   0.25 ],  # R8  35–40%   score=3.5  (cat=medium)

    # ── MEDIUM: approaching threshold ───────────────────────────────────────────────
    [8000.00,    1,   0.20,   0.10,   22,     0,    0.55,    1,    0.30,   0.25 ],  # R9  40–45%   score=5.0  (+counterfeit)
    [8000.00,    1,   0.20,   0.10,   22,     0,    0.55,    1,    0.30,   0.55 ],  # R10 45–50%   score=6.0  (+discount)

    # ── MEDIUM-HIGH: just over threshold ────────────────────────────────────────────
 
    [8000.00,    2,   0.20,   0.15,   22,     0,    0.55,    1,    0.30,   0.55 ],  # R12 55–60%   score=9.5  (+cat=high)

    # ── HIGH: stacking major flags ───────────────────────────────────────────────────

    [8000.00,    1,   0.27,   0.17,   18,     0,    0.55,    1,    0.65,   0.55 ],  # R14 65–70%   score=13.0 (+return_rate)

    # ── HIGH: adding price trigger ───────────────────────────────────────────────────
    [16000.00,   0,   0.32,   0.16,   16,     0,    0.55,    1,    0.65,   0.55 ],  # R15 70–75%   score=15.0 (+price>15000)
    [7000.00,   1,   0.21,   0.19,   12,     0,    0.49,    1,    0.68,   0.55 ],  # R16 75–80%   score=15.0

    # ── VERY HIGH: compounding all heavy flags ───────────────────────────────────────
    [20000.00,   2,   0.50,   0.30,   24,     0,    0.65,    1,    0.72,   0.58 ],  # R17 80–85%   score=15.0
    [2000.00,    1,   0.21,   0.11,   18,     0,    0.51,    1,    0.65,   0.55 ],  # R14 65–70%   score=13.0 (+return_rate)
]

TARGET_BANDS = [
    "0–5%",   "5–10%",  "10–15%", "15–20%", "20–25%",
    "25–30%", "30–35%", "35–40%", "40–45%", "45–50%",
    "50–55%", "55–60%", "60–65%", "65–70%", "70–75%",
    "75–80%", "80–85%", "85–90%", "90–95%", "95–100%",
]

# Reference raw scores per row
RAW_SCORES = [0.0, 0.0, 0.0, 0.0,
              1.0, 2.0,
              3.5, 3.5,
              5.0, 6.0,
              8.0, 9.5,
              11.0,13.0,
              15.0,15.0,
              15.0,15.0,15.0,15.0]

# ═══════════════════════════════════════════════════════════════════
# 3. PREPARE, SCALE & PREDICT
# ═══════════════════════════════════════════════════════════════════
input_df     = pd.DataFrame(samples, columns=features)
input_scaled = scaler.transform(input_df)

predicted_labels        = model.predict(input_scaled)
predicted_probabilities = model.predict_proba(input_scaled)[:, 1]

# ═══════════════════════════════════════════════════════════════════
# 4. DISPLAY RESULTS — MAIN TABLE
# ═══════════════════════════════════════════════════════════════════
print("=" * 82)
print("        📦 PRODUCT RISK PREDICTION — 20 TEST SAMPLES")
print("=" * 82)
print(f"  {'#':<4} {'Target Band':<13} {'Risk %':<10} {'Visual Bar':<26} {'Label'}")
print("-" * 82)

for i,(prob,label,band) in enumerate(zip(predicted_probabilities,predicted_labels,TARGET_BANDS)):
    status = "🔴 HIGH RISK (1)" if label == 1 else "🟢 LOW RISK  (0)"
    bar    = "█" * int(prob * 25)
    print(f"  {i+1:<4} {band:<13} {prob*100:>6.2f}%   {bar:<26} {status}")

print("=" * 82)

# ═══════════════════════════════════════════════════════════════════
# 5. SUMMARY STATS
# ═══════════════════════════════════════════════════════════════════
print(f"\n  📊 Probability Range   : {predicted_probabilities.min()*100:.2f}% → {predicted_probabilities.max()*100:.2f}%")
print(f"  📊 High Risk (1) count : {predicted_labels.sum()} / 20")
print(f"  📊 Low  Risk (0) count : {(predicted_labels==0).sum()} / 20")
spread_ok = predicted_probabilities.max() > 0.90 and predicted_probabilities.min() < 0.10
print(f"  📊 Full 0–100% Spread  : {'✅ YES' if spread_ok else '⚠️  Partial — retrain with more noise'}")

# ═══════════════════════════════════════════════════════════════════
# 6. DETAILED FLAG TABLE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*105)
print("  📋 FEATURE DETAILS & ACTIVE FLAGS PER ROW")
print("="*105)
print(f"  {'#':<3} {'price':>8} {'cat':>4} {'ret':>6} {'fraud':>6} {'days':>5} {'ser':>4} {'ctft':>6} {'frag':>5} {'sell':>6} {'disc':>6} {'score':>7}  flags")
print("  " + "-"*100)

flag_checks = [
    ("price",  lambda r: r[0] > 15000),
    ("cat=2",  lambda r: r[1] == 2),
    ("ret",    lambda r: r[2] > 0.35),
    ("fraud",  lambda r: r[3] > 0.20),
    ("days",   lambda r: r[4] > 20),
    ("noser",  lambda r: r[5] == 0),
    ("ctft",   lambda r: r[6] > 0.5),
    ("frag",   lambda r: r[7] == 1),
    ("sell",   lambda r: r[8] > 0.6),
    ("disc",   lambda r: r[9] > 0.5),
]

for i,row in enumerate(samples):
    active = [name for name,check in flag_checks if check(row)]
    flag_str = ",".join(active) if active else "none"
    p,c,rr,fr,d,s,ct,f,sp,dp = row
    print(f"  {i+1:<3} {p:>8.0f} {c:>4} {rr:>6.2f} {fr:>6.2f} {d:>5} {s:>4} {ct:>6.2f} {f:>5} {sp:>6.2f} {dp:>6.2f} {RAW_SCORES[i]:>7.1f}  [{flag_str}]")

print("="*105)
print("\n  product_category: 0=Low Risk  1=Medium Risk  2=High Risk")
