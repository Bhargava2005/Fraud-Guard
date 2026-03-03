import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, precision_score,
    recall_score, brier_score_loss, confusion_matrix,
    precision_recall_curve, average_precision_score
)

# ═══════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════
data = pd.read_csv("../assets/order_risk_data.csv")

y = data["order_risk_label"]
X = data.drop("order_risk_label", axis=1)

# ✅ BUG FIX 1: Original saved data.columns BEFORE drop
#    → "order_risk_label" was included in the features list
#    → prediction code would crash or produce wrong results
feature_names = X.columns.tolist()

# ═══════════════════════════════════════════════════════════════════
# 2. SCALE + SPLIT
# ═══════════════════════════════════════════════════════════════════
# ✅ BUG FIX 2: Original overwrote X with numpy array from scaler
#    X = scaler.fit_transform(X)  ← X becomes ndarray, feature names lost
#    Fixed: keep X as DataFrame, use X_scaled separately
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, stratify=y, test_size=0.2, random_state=42
)

# ═══════════════════════════════════════════════════════════════════
# 3. WHY LightGBM IS BEST FOR ORDER RISK DATA
# ═══════════════════════════════════════════════════════════════════
# Order risk data has a very specific feature structure:
#
#   → High-value continuous : order_value (300–40000), checkout_time_sec (10–610)
#   → Low-range continuous  : customer_tenure_days (1–1500)
#   → Small integer counts  : item_quantity(1–10), identical_items(0–10),
#                             failed_payment_attempts(0–4), order_velocity(1–8)
#   → Time-based            : order_hour (0–23)
#   → Binary flags          : payment_method, address_mismatch
#
# WHY LightGBM wins for this specific mix:
#   ✔ order_value has extreme range (300 to 40300) — LightGBM's histogram-based
#     binning handles wide-range features efficiently without being misled by outliers
#   ✔ Interaction patterns matter: (high order_value) AND (fast checkout) AND
#     (address_mismatch) is a specific fraud signature — LightGBM's leaf-wise
#     growth finds this combination as a single decision path
#   ✔ GradientBoosting is too slow on 50k records for panel demo
#   ✔ XGBoost is competitive but LightGBM trains 3–5x faster with similar accuracy
#   ✔ class_weight="balanced" natively handles fraud minority class

# ═══════════════════════════════════════════════════════════════════
# 4. DEFINE 3 MODELS
# ═══════════════════════════════════════════════════════════════════
model_defs = {
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, learning_rate=0.05,
        scale_pos_weight=(y==0).sum()/(y==1).sum(),
        random_state=42, eval_metric="logloss", verbosity=0
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300, learning_rate=0.05,
        class_weight="balanced", random_state=42, verbose=-1
    ),
}

COLORS = {
    "Gradient Boosting": "#F97316",   # orange
    "XGBoost"          : "#3B82F6",   # blue
    "LightGBM"         : "#10B981",   # green ← chosen
}

# ═══════════════════════════════════════════════════════════════════
# 5. TRAIN, CALIBRATE & EVALUATE
# ═══════════════════════════════════════════════════════════════════
# ✅ BUG FIX 3: Original had NO CalibratedClassifierCV
#    Raw LightGBM probabilities are extreme (0.00001 / 0.99999)
#    CalibratedClassifierCV with isotonic regression maps them to
#    true probabilities (e.g. 0.62 means 62% chance of fraud)
print("⏳ Training models...")
trained = {}
for name, base in model_defs.items():
    print(f"   Training {name}...")
    cal     = CalibratedClassifierCV(base, method='isotonic', cv=5)
    cal.fit(X_train, y_train)
    yp      = cal.predict(X_test)
    ypr     = cal.predict_proba(X_test)[:, 1]
    trained[name] = dict(
        model=cal, y_pred=yp, y_proba=ypr,
        roc_auc=roc_auc_score(y_test, ypr),
        f1=f1_score(y_test, yp),
        precision=precision_score(y_test, yp),
        recall=recall_score(y_test, yp),
        brier=brier_score_loss(y_test, ypr),
        ap=average_precision_score(y_test, ypr),
        cm=confusion_matrix(y_test, yp)
    )
print("✅ All models trained!\n")

names  = list(trained.keys())
colors = [COLORS[n] for n in names]

# ═══════════════════════════════════════════════════════════════════
# 6. SAVE LightGBM + ARTIFACTS
# ═══════════════════════════════════════════════════════════════════
joblib.dump(trained["LightGBM"]["model"], "../trained_models/order_risk_models/order_risk_model.pkl")
joblib.dump(scaler,                       "../trained_models/order_risk_models/order_risk_scaler.pkl")
joblib.dump(feature_names,                "../trained_models/order_risk_models/order_risk_features.pkl")
print("✅ LightGBM model + scaler + features saved.\n")

# ═══════════════════════════════════════════════════════════════════
# 7. VISUAL STYLE
# ═══════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor": "#0F172A", "axes.facecolor"  : "#1E293B",
    "axes.edgecolor"  : "#334155", "axes.labelcolor" : "#CBD5E1",
    "axes.titlecolor" : "#F1F5F9", "xtick.color"     : "#94A3B8",
    "ytick.color"     : "#94A3B8", "grid.color"      : "#334155",
    "grid.linewidth"  : 0.6,       "text.color"      : "#F1F5F9",
    "font.family"     : "monospace",
    "axes.spines.top" : False,     "axes.spines.right": False,
})

# ═══════════════════════════════════════════════════════════════════
# 8. FIGURE 1 — MAIN PERFORMANCE DASHBOARD
# ═══════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(20, 16), facecolor="#0F172A")
fig1.suptitle("ORDER FRAUD RISK MODEL — PERFORMANCE COMPARISON",
              fontsize=18, fontweight="bold", color="#F1F5F9", y=0.97)
gs = gridspec.GridSpec(3, 3, figure=fig1, hspace=0.45, wspace=0.35)

# ROC Curve
ax1 = fig1.add_subplot(gs[0, :2])
ax1.set_title("ROC CURVE  (higher = better)", fontsize=11, pad=10, color="#94A3B8")
for name in names:
    fpr, tpr, _ = roc_curve(y_test, trained[name]["y_proba"])
    lw = 3 if name == "LightGBM" else 1.8
    ax1.plot(fpr, tpr, color=COLORS[name], linewidth=lw,
             label=f"{name}  (AUC={trained[name]['roc_auc']:.4f})")
ax1.plot([0,1],[0,1], ":", color="#475569", linewidth=1)
ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
ax1.legend(loc="lower right", fontsize=9, framealpha=0.2)
ax1.grid(True, alpha=0.3); ax1.set_xlim([0,1]); ax1.set_ylim([0,1.02])

# Metric Card
ax2 = fig1.add_subplot(gs[0, 2]); ax2.axis("off")
ax2.set_title("METRIC SUMMARY", fontsize=11, pad=10, color="#94A3B8")
mkeys = ["roc_auc","f1","precision","recall"]
mlbls = ["ROC-AUC","F1","Precision","Recall"]
col_x = [0.05, 0.38, 0.65, 0.88]
for j,(ml,cx) in enumerate(zip(mlbls,col_x)):
    ax2.text(cx, 0.92, ml, ha="left", fontsize=7.5, color="#64748B", transform=ax2.transAxes)
for i, name in enumerate(names):
    yp = 0.85 - i*0.27
    ax2.add_patch(FancyBboxPatch((0.0,yp-0.08),1.0,0.22,
        boxstyle="round,pad=0.02",linewidth=1.2,
        edgecolor=COLORS[name],facecolor=COLORS[name]+"22",transform=ax2.transAxes))
    ax2.text(0.05,yp+0.04,name,ha="left",fontsize=8.5,fontweight="bold",
             color=COLORS[name],transform=ax2.transAxes)
    for j,(mk,cx) in enumerate(zip(mkeys,col_x)):
        ax2.text(cx,yp-0.04,f"{trained[name][mk]:.3f}",ha="left",
                 fontsize=9,color="#E2E8F0",transform=ax2.transAxes)

# Grouped Bar
ax3 = fig1.add_subplot(gs[1, 0])
x = np.arange(len(mlbls)); width = 0.25
for i,name in enumerate(names):
    vals = [trained[name][mk] for mk in mkeys]
    bars = ax3.bar(x+i*width,vals,width,color=COLORS[name],alpha=0.85,
                   label=name,edgecolor="#0F172A",linewidth=0.5)
    for bar,val in zip(bars,vals):
        ax3.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,
                 f"{val:.2f}",ha="center",va="bottom",fontsize=6.5,color="#CBD5E1")
ax3.set_title("METRICS COMPARISON",fontsize=11,pad=10,color="#94A3B8")
ax3.set_xticks(x+width); ax3.set_xticklabels(mlbls,fontsize=8)
ax3.set_ylim([0.6,1.05]); ax3.legend(fontsize=7,framealpha=0.2)
ax3.grid(True,axis="y",alpha=0.3)

# Brier Score
ax4 = fig1.add_subplot(gs[1, 1])
bvals = [trained[n]["brier"] for n in names]
bars  = ax4.barh(names,bvals,color=colors,alpha=0.85,
                 edgecolor="#0F172A",linewidth=0.5,height=0.5)
for bar,val in zip(bars,bvals):
    ax4.text(val+0.001,bar.get_y()+bar.get_height()/2,
             f"{val:.4f}",va="center",fontsize=9,color="#E2E8F0")
ax4.set_title("BRIER SCORE  (lower = better)",fontsize=11,pad=10,color="#94A3B8")
ax4.set_xlabel("Brier Score"); ax4.grid(True,axis="x",alpha=0.3); ax4.invert_xaxis()

# Precision-Recall Curve
ax5 = fig1.add_subplot(gs[1, 2])
ax5.set_title("PRECISION-RECALL CURVE",fontsize=11,pad=10,color="#94A3B8")
for name in names:
    prec,rec,_ = precision_recall_curve(y_test,trained[name]["y_proba"])
    lw = 3 if name == "LightGBM" else 1.8
    ax5.plot(rec,prec,color=COLORS[name],linewidth=lw,
             label=f"{name}  (AP={trained[name]['ap']:.3f})")
ax5.set_xlabel("Recall"); ax5.set_ylabel("Precision")
ax5.legend(fontsize=8,framealpha=0.2); ax5.grid(True,alpha=0.3)
ax5.set_xlim([0,1]); ax5.set_ylim([0,1.02])

# Confusion Matrices
for i,name in enumerate(names):
    ax  = fig1.add_subplot(gs[2,i])
    cm  = trained[name]["cm"]
    ax.imshow(cm,cmap="Blues",aspect="auto")
    ax.set_title(f"CONFUSION MATRIX\n{name}",fontsize=9,pad=8,color="#94A3B8")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Low\nRisk(0)","High\nRisk(1)"],fontsize=8)
    ax.set_yticklabels(["Low\nRisk(0)","High\nRisk(1)"],fontsize=8)
    ax.set_xlabel("Predicted",fontsize=8); ax.set_ylabel("Actual",fontsize=8)
    for r in range(2):
        for c in range(2):
            v = cm[r,c]
            ax.text(c,r,f"{v:,}\n({v/cm.sum()*100:.1f}%)",
                    ha="center",va="center",fontsize=9,fontweight="bold",
                    color="white" if v>cm.max()/2 else "#1E293B")
    if name == "LightGBM":
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS["LightGBM"]); spine.set_linewidth(2.5); spine.set_visible(True)

plt.savefig("../trained_models/order_risk_models/fig1_performance_dashboard.png",
            dpi=150, bbox_inches="tight", facecolor="#0F172A")
print("📊 Figure 1 saved: fig1_performance_dashboard.png")

# ═══════════════════════════════════════════════════════════════════
# 9. FIGURE 2 — PROBABILITY DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(1,3,figsize=(20,6),facecolor="#0F172A")
fig2.suptitle("PROBABILITY CALIBRATION & DISTRIBUTION ANALYSIS",
              fontsize=16,fontweight="bold",color="#F1F5F9",y=1.02)
for ax,name in zip(axes,names):
    proba = trained[name]["y_proba"]
    ax.hist(proba[y_test==0],bins=40,alpha=0.6,color="#3B82F6",label="Low Risk (0)",density=True)
    ax.hist(proba[y_test==1],bins=40,alpha=0.6,color="#EF4444",label="High Risk (1)",density=True)
    ax.axvline(0.5,color="#FBBF24",linestyle="--",linewidth=1.5,label="Threshold (0.5)")
    ax.set_title(f"{name}\nROC-AUC: {trained[name]['roc_auc']:.4f}",
                 fontsize=12,color=COLORS[name],fontweight="bold",pad=10)
    ax.set_xlabel("Predicted Risk Probability",fontsize=9)
    ax.set_ylabel("Density",fontsize=9)
    ax.legend(fontsize=8,framealpha=0.2); ax.grid(True,alpha=0.3)
    if name == "LightGBM":
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS["LightGBM"]); spine.set_linewidth(2.5); spine.set_visible(True)
plt.tight_layout()
plt.savefig("../trained_models/order_risk_models/fig2_probability_distribution.png",
            dpi=150, bbox_inches="tight", facecolor="#0F172A")
print("📊 Figure 2 saved: fig2_probability_distribution.png")

# ═══════════════════════════════════════════════════════════════════
# 10. FIGURE 3 — FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════
fig3, ax = plt.subplots(figsize=(12,7),facecolor="#0F172A")
fig3.suptitle("LIGHTGBM — ORDER RISK FEATURE IMPORTANCE",
              fontsize=14,fontweight="bold",color="#F1F5F9")
lgbm_base   = trained["LightGBM"]["model"].calibrated_classifiers_[0].estimator
imps        = lgbm_base.feature_importances_
sidx        = np.argsort(imps)
simp        = imps[sidx]
sfeat       = [feature_names[i] for i in sidx]
bcols       = ["#EF4444" if v==max(imps) else
               "#F97316" if v>=np.percentile(imps,75) else
               "#10B981" for v in simp]
bars = ax.barh(sfeat,simp,color=bcols,edgecolor="#0F172A",linewidth=0.5,height=0.6)
for bar,val in zip(bars,simp):
    ax.text(val+max(imps)*0.01,bar.get_y()+bar.get_height()/2,
            f"{val}",va="center",fontsize=9,color="#E2E8F0")
ax.set_xlabel("Feature Importance Score",fontsize=10)
ax.grid(True,axis="x",alpha=0.3)
ax.legend(handles=[
    mpatches.Patch(color="#EF4444",label="Highest importance"),
    mpatches.Patch(color="#F97316",label="Top 25%"),
    mpatches.Patch(color="#10B981",label="Standard"),
],fontsize=9,framealpha=0.2,loc="lower right")
plt.tight_layout()
plt.savefig("../trained_models/order_risk_models/fig3_feature_importance.png",
            dpi=150, bbox_inches="tight", facecolor="#0F172A")
print("📊 Figure 3 saved: fig3_feature_importance.png")

# ═══════════════════════════════════════════════════════════════════
# 11. FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  📈 ORDER RISK — MODEL COMPARISON SUMMARY")
print("="*65)
print(f"  {'Model':<22} {'ROC-AUC':>8} {'F1':>7} {'Brier':>8}")
print("-"*65)
for name in names:
    marker = "  ← SELECTED" if name == "LightGBM" else ""
    print(f"  {name:<22} {trained[name]['roc_auc']:>8.4f} "
          f"{trained[name]['f1']:>7.4f} {trained[name]['brier']:>8.4f}{marker}")
print("="*65)
print("""
  🎓 WHY LightGBM FOR ORDER RISK:
  ✔ order_value range (300–40300) handled by histogram binning
  ✔ Interaction: high_value + fast_checkout + address_mismatch
    detected as single leaf path (leaf-wise growth)
  ✔ 3–5x faster than GradientBoosting on 50k records
  ✔ class_weight=balanced for fraud minority class
  ✔ Paired with CalibratedClassifierCV for true probabilities
""")
