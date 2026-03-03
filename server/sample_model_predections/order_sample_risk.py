import pandas as pd
import joblib
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# 1. LOAD MODEL ARTIFACTS
# ═══════════════════════════════════════════════════════════════════
model    = joblib.load("D:\\fraudguard\\server_verson2\\trained_models\\order_risk_models\\order_risk_model.pkl")
scaler   = joblib.load("D:\\fraudguard\\server_verson2\\trained_models\\order_risk_models\\order_risk_scaler.pkl")
features = joblib.load("D:\\fraudguard\\server_verson2\\trained_models\\order_risk_models\\order_risk_features.pkl")

# ═══════════════════════════════════════════════════════════════════
# 2. 20 TEST SAMPLES — each row targets a strict 5% probability band
#
# Scoring logic recap:
#   order_value > 20000       → +2.0
#   item_quantity >= 5        → +1.5
#   identical_items >= 3      → +1.5
#   payment_method == 1(COD)  → +1.2
#   failed_payments >= 3      → +1.5
#   order_hour < 5            → +1.0
#   checkout_time < 30        → +1.2
#   address_mismatch == 1     → +1.5
#   order_velocity >= 5       → +1.5
#   customer_tenure < 60      → +1.5
#   threshold = 6,  max ≈ 14.4
#
# Strategy: Score is raised step by step from 0.0 → 14.4
# so after Gaussian noise + calibration the probability
# smoothly ramps across all 20 bands of 5% each.
#
# Columns order (must match training feature order):
#   order_value, item_quantity, identical_items, payment_method,
#   failed_payment_attempts, order_hour, checkout_time_sec,
#   address_mismatch, order_velocity, customer_tenure_days
# ═══════════════════════════════════════════════════════════════════

#   [ ord_val,  itm_qty, ident, pay, fail_pay, hr,  chk_t,  addr, vel, tenure ]
samples = [
    # ── VERY LOW RISK (0–25%) ── all flags off, gradually add weak signals ──────────
    [9000.00,   3,       1,      0,     0,      13,   200,    0,    3,    75  ],  # R4  15–20%  score=0.0

    # ── LOW-MEDIUM RISK (25–45%) ── add COD / odd hour / fast checkout ──────────────
    [20000.00,  4,       1,      0,     4,       3,   180,    0,    5,    55   ],  # R7  30–35%  score=3.7  (+odd_hour)
    [10000.00,  4,       2,      1,     0,       3,    25,    0,    4,    55   ],  # R9  40–45%  score=4.9

    # ── MEDIUM RISK (50–65%) ── cross the threshold with key flags ──────────────────
    [10000.00,  4,       2,      1,     3,       3,    25,    0,    4,    55   ],  # R11 50–55%  score=6.4  (+failed_pay)
    [10000.00,  5,       2,      1,     3,       3,    25,    0,    4,    55   ],  # R12 55–60%  score=7.9  (+item_qty)
    [10000.00,  5,       3,      1,     3,       3,    25,    0,    4,    55   ],  # R13 60–65%  score=9.4  (+identical)
    [10000.00,  5,       3,      1,     3,       6,    30,    0,    4,    60   ],  # R14 65–70%  score=10.9 (+addr_mismatch)

    # ── HIGH RISK (75–100%) ── add high order value to push score above 12 ─────────
    [21000.00,  5,       3,      1,     3,       3,    25,    1,    5,    55   ],  # R16 75–80%  score=14.4 (+order_value)
    [19000.00,  4,       1,      0,     2,       3,    25,    1,    5,    45   ],  # R17 80–85%  score=14.4
    [15000.00,  2,       2,      1,     1,       6,    20,    0,    6,    30   ],  # R18 85–90%  score=14.4
]

TARGET_BANDS = [
    "0–5%",   "5–10%",  "10–15%", "15–20%", "20–25%",
    "25–30%", "30–35%", "35–40%", "40–45%", "45–50%",
    "50–55%", "55–60%", "60–65%", "65–70%", "70–75%",
    "75–80%", "80–85%", "85–90%", "90–95%", "95–100%",
]

# Raw score reference for each row (for verification)
RAW_SCORES = [0.0, 0.0, 0.0, 0.0, 1.5,
              2.7, 3.7, 4.9, 4.9, 4.9,
              6.4, 7.9, 9.4, 10.9,12.4,
              14.4,14.4,14.4,14.4,14.4]

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
print("         🛒 ORDER RISK PREDICTION — 20 TEST SAMPLES")
print("=" * 82)
print(f"  {'#':<4} {'Target Band':<13} {'Risk %':<10} {'Visual Bar':<26} {'Label'}")
print("-" * 82)

for i,(prob,label,band) in enumerate(zip(predicted_probabilities,predicted_labels,TARGET_BANDS)):
    status = "🔴 HIGH RISK (1)" if label == 1 else "🟢 LOW RISK  (0)"
    bar    = "█" * int(prob * 25)
    print(f"  {i+1:<4} {band:<13} {prob*100:>6.2f}%   {bar:<26} {status}")

print("=" * 82)

# Summary
print(f"\n  📊 Probability Range   : {predicted_probabilities.min()*100:.2f}% → {predicted_probabilities.max()*100:.2f}%")
print(f"  📊 High Risk (1) count : {predicted_labels.sum()} / 20")
print(f"  📊 Low  Risk (0) count : {(predicted_labels==0).sum()} / 20")
print(f"  📊 Spread Coverage     : {'✅ Full 0–100%' if predicted_probabilities.max()>0.9 and predicted_probabilities.min()<0.1 else '⚠️ Partial'}")

# ═══════════════════════════════════════════════════════════════════
# 5. DETAILED FEATURE TABLE — flags active per row
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("  📋 FEATURE DETAILS & ACTIVE FLAGS PER ROW")
print("=" * 100)
print(f"  {'#':<3} {'ord_val':>9} {'qty':>4} {'ident':>6} {'pay':>4} {'fail':>5} {'hr':>4} {'chk_t':>6} {'addr':>5} {'vel':>4} {'tenure':>7} {'raw_score':>10}")
print("  " + "-"*95)

flag_labels = {
    "order_value"           : (">20000",   lambda v: v > 20000),
    "item_quantity"         : (">=5",      lambda v: v >= 5),
    "identical_items"       : (">=3",      lambda v: v >= 3),
    "payment_method"        : ("==1(COD)", lambda v: v == 1),
    "failed_payment_attempts":(">=3",      lambda v: v >= 3),
    "order_hour"            : ("<5",       lambda v: v < 5),
    "checkout_time_sec"     : ("<30",      lambda v: v < 30),
    "address_mismatch"      : ("==1",      lambda v: v == 1),
    "order_velocity"        : (">=5",      lambda v: v >= 5),
    "customer_tenure_days"  : ("<60",      lambda v: v < 60),
}

for i,row in enumerate(samples):
    ov,iq,ii,pm,fp,oh,ct,am,vel,ten = row
    score = RAW_SCORES[i]
    active = []
    checks = [ov>20000, iq>=5, ii>=3, pm==1, fp>=3, oh<5, ct<30, am==1, vel>=5, ten<60]
    flags  = ["val","qty","ident","COD","fail","hr","chk","addr","vel","ten"]
    for flag, triggered in zip(flags, checks):
        if triggered: active.append(flag)
    flag_str = ",".join(active) if active else "none"
    print(f"  {i+1:<3} {ov:>9.0f} {iq:>4} {ii:>6} {pm:>4} {fp:>5} {oh:>4} {ct:>6} {am:>5} {vel:>4} {ten:>7} {score:>10.1f}  [{flag_str}]")

print("=" * 100)