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
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, precision_score,
    recall_score, brier_score_loss, confusion_matrix,
    classification_report, precision_recall_curve, average_precision_score
)

# ═══════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════
data = pd.read_csv("../assets/device_risk_data.csv")

y = data["device_risk_label"]
X = data.drop("device_risk_label", axis=1)

# ═══════════════════════════════════════════════════════════════════
# 2. ANOMALY SCORE FEATURE (IsolationForest)
# ═══════════════════════════════════════════════════════════════════
# IsolationForest adds an extra engineered feature — anomaly_score —
# which captures unusual device behavior patterns that rule-based
# scoring might miss. This is a hybrid approach: rule-based labels
# + unsupervised anomaly detection as an input feature.
iso = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
X = X.copy()
X["anomaly_score"] = iso.fit_predict(X)   # -1 = anomalous, 1 = normal

feature_names = X.columns.tolist()

# ═══════════════════════════════════════════════════════════════════
# 3. SCALE + SPLIT
# ═══════════════════════════════════════════════════════════════════
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, stratify=y, test_size=0.2, random_state=42
)

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
    "Gradient Boosting": "#F97316",
    "XGBoost"          : "#3B82F6",
    "LightGBM"         : "#10B981",
}

# ═══════════════════════════════════════════════════════════════════
# 5. TRAIN, CALIBRATE & EVALUATE ALL 3 MODELS
# ═══════════════════════════════════════════════════════════════════
print("⏳ Training models...")
trained = {}
for name, base in model_defs.items():
    print(f"   Training {name}...")
    cal     = CalibratedClassifierCV(base, method='isotonic', cv=5)
    cal.fit(X_train, y_train)
    y_pred  = cal.predict(X_test)
    y_proba = cal.predict_proba(X_test)[:, 1]
    trained[name] = {
        "model"    : cal,
        "y_pred"   : y_pred,
        "y_proba"  : y_proba,
        "roc_auc"  : roc_auc_score(y_test, y_proba),
        "f1"       : f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall"   : recall_score(y_test, y_pred),
        "brier"    : brier_score_loss(y_test, y_proba),
        "ap"       : average_precision_score(y_test, y_proba),
        "cm"       : confusion_matrix(y_test, y_pred),
    }
print("✅ All models trained!\n")

names  = list(trained.keys())
colors = [COLORS[n] for n in names]

# ═══════════════════════════════════════════════════════════════════
# 6. SAVE BEST (LightGBM) MODEL
# ═══════════════════════════════════════════════════════════════════
joblib.dump(trained["LightGBM"]["model"], "../trained_models/device_risk_models/device_risk_model.pkl")
joblib.dump(scaler,                       "../trained_models/device_risk_models/device_risk_scaler.pkl")
joblib.dump(iso,                          "../trained_models/device_risk_models/device_risk_isolation.pkl")
joblib.dump(feature_names,                "../trained_models/device_risk_models/device_risk_features.pkl")
print("✅ LightGBM model + scaler + IsolationForest saved.\n")

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
fig1.suptitle("DEVICE FRAUD RISK MODEL — PERFORMANCE COMPARISON",
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

# Metric Summary Card
ax2 = fig1.add_subplot(gs[0, 2]); ax2.axis("off")
ax2.set_title("METRIC SUMMARY", fontsize=11, pad=10, color="#94A3B8")
metric_keys = ["roc_auc","f1","precision","recall"]
metric_lbls = ["ROC-AUC","F1","Precision","Recall"]
col_x = [0.05, 0.38, 0.65, 0.88]
for j, (ml, cx) in enumerate(zip(metric_lbls, col_x)):
    ax2.text(cx, 0.92, ml, ha="left", fontsize=7.5, color="#64748B", transform=ax2.transAxes)
for i, name in enumerate(names):
    y_pos = 0.85 - i*0.27
    ax2.add_patch(FancyBboxPatch((0.0, y_pos-0.08), 1.0, 0.22,
        boxstyle="round,pad=0.02", linewidth=1.2,
        edgecolor=COLORS[name], facecolor=COLORS[name]+"22", transform=ax2.transAxes))
    ax2.text(0.05, y_pos+0.04, name, ha="left", fontsize=8.5, fontweight="bold",
             color=COLORS[name], transform=ax2.transAxes)
    for j, (mk, cx) in enumerate(zip(metric_keys, col_x)):
        ax2.text(cx, y_pos-0.04, f"{trained[name][mk]:.3f}", ha="left",
                 fontsize=9, color="#E2E8F0", transform=ax2.transAxes)

# Grouped Bar Chart
ax3 = fig1.add_subplot(gs[1, 0])
x = np.arange(len(metric_lbls)); width = 0.25
for i, name in enumerate(names):
    vals = [trained[name][mk] for mk in metric_keys]
    bars = ax3.bar(x+i*width, vals, width, color=COLORS[name], alpha=0.85,
                   label=name, edgecolor="#0F172A", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=6.5, color="#CBD5E1")
ax3.set_title("METRICS COMPARISON", fontsize=11, pad=10, color="#94A3B8")
ax3.set_xticks(x+width); ax3.set_xticklabels(metric_lbls, fontsize=8)
ax3.set_ylim([0.6, 1.05]); ax3.legend(fontsize=7, framealpha=0.2)
ax3.grid(True, axis="y", alpha=0.3)

# Brier Score
ax4 = fig1.add_subplot(gs[1, 1])
brier_vals = [trained[n]["brier"] for n in names]
bars = ax4.barh(names, brier_vals, color=colors, alpha=0.85,
                edgecolor="#0F172A", linewidth=0.5, height=0.5)
for bar, val in zip(bars, brier_vals):
    ax4.text(val+0.001, bar.get_y()+bar.get_height()/2,
             f"{val:.4f}", va="center", fontsize=9, color="#E2E8F0")
ax4.set_title("BRIER SCORE  (lower = better)", fontsize=11, pad=10, color="#94A3B8")
ax4.set_xlabel("Brier Score"); ax4.grid(True, axis="x", alpha=0.3); ax4.invert_xaxis()

# Precision-Recall Curve
ax5 = fig1.add_subplot(gs[1, 2])
ax5.set_title("PRECISION-RECALL CURVE", fontsize=11, pad=10, color="#94A3B8")
for name in names:
    prec, rec, _ = precision_recall_curve(y_test, trained[name]["y_proba"])
    lw = 3 if name == "LightGBM" else 1.8
    ax5.plot(rec, prec, color=COLORS[name], linewidth=lw,
             label=f"{name}  (AP={trained[name]['ap']:.3f})")
ax5.set_xlabel("Recall"); ax5.set_ylabel("Precision")
ax5.legend(fontsize=8, framealpha=0.2); ax5.grid(True, alpha=0.3)
ax5.set_xlim([0,1]); ax5.set_ylim([0,1.02])

# Confusion Matrices
for i, name in enumerate(names):
    ax = fig1.add_subplot(gs[2, i])
    cm = trained[name]["cm"]
    ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_title(f"CONFUSION MATRIX\n{name}", fontsize=9, pad=8, color="#94A3B8")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Low\nRisk(0)","High\nRisk(1)"], fontsize=8)
    ax.set_yticklabels(["Low\nRisk(0)","High\nRisk(1)"], fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("Actual", fontsize=8)
    for r in range(2):
        for c in range(2):
            val = cm[r,c]
            ax.text(c, r, f"{val:,}\n({val/cm.sum()*100:.1f}%)",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if cm[r,c]>cm.max()/2 else "#1E293B")
    if name == "LightGBM":
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS["LightGBM"]); spine.set_linewidth(2.5); spine.set_visible(True)

plt.savefig("../trained_models/device_risk_models/fig1_performance_dashboard.png",
            dpi=150, bbox_inches="tight", facecolor="#0F172A")
print("📊 Figure 1 saved: fig1_performance_dashboard.png")

# ═══════════════════════════════════════════════════════════════════
# 9. FIGURE 2 — PROBABILITY DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor="#0F172A")
fig2.suptitle("PROBABILITY CALIBRATION & DISTRIBUTION ANALYSIS",
              fontsize=16, fontweight="bold", color="#F1F5F9", y=1.02)
for ax, name in zip(axes, names):
    proba = trained[name]["y_proba"]
    ax.hist(proba[y_test==0], bins=40, alpha=0.6, color="#3B82F6",
            label="Low Risk (0)", density=True)
    ax.hist(proba[y_test==1], bins=40, alpha=0.6, color="#EF4444",
            label="High Risk (1)", density=True)
    ax.axvline(0.5, color="#FBBF24", linestyle="--", linewidth=1.5, label="Threshold (0.5)")
    ax.set_title(f"{name}\nROC-AUC: {trained[name]['roc_auc']:.4f}",
                 fontsize=12, color=COLORS[name], fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Risk Probability", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.2); ax.grid(True, alpha=0.3)
    if name == "LightGBM":
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS["LightGBM"]); spine.set_linewidth(2.5); spine.set_visible(True)
plt.tight_layout()
plt.savefig("../trained_models/device_risk_models/fig2_probability_distribution.png",
            dpi=150, bbox_inches="tight", facecolor="#0F172A")
print("📊 Figure 2 saved: fig2_probability_distribution.png")

# ═══════════════════════════════════════════════════════════════════
# 10. FIGURE 3 — FEATURE IMPORTANCE (LightGBM)
# ═══════════════════════════════════════════════════════════════════
fig3, ax = plt.subplots(figsize=(12, 7), facecolor="#0F172A")
fig3.suptitle("LIGHTGBM — DEVICE RISK FEATURE IMPORTANCE",
              fontsize=14, fontweight="bold", color="#F1F5F9")
lgbm_base    = trained["LightGBM"]["model"].calibrated_classifiers_[0].estimator
importances  = lgbm_base.feature_importances_
sorted_idx   = np.argsort(importances)
sorted_imp   = importances[sorted_idx]
sorted_feat  = [feature_names[i] for i in sorted_idx]
bar_colors   = ["#EF4444" if imp==max(importances) else
                "#F97316" if imp>=np.percentile(importances,75) else
                "#10B981" for imp in sorted_imp]
bars = ax.barh(sorted_feat, sorted_imp, color=bar_colors,
               edgecolor="#0F172A", linewidth=0.5, height=0.6)
for bar, val in zip(bars, sorted_imp):
    ax.text(val+max(importances)*0.01, bar.get_y()+bar.get_height()/2,
            f"{val}", va="center", fontsize=9, color="#E2E8F0")
ax.set_xlabel("Feature Importance Score", fontsize=10)
ax.grid(True, axis="x", alpha=0.3)
legend_patches = [
    mpatches.Patch(color="#EF4444", label="Highest importance"),
    mpatches.Patch(color="#F97316", label="Top 25%"),
    mpatches.Patch(color="#10B981", label="Standard"),
]
ax.legend(handles=legend_patches, fontsize=9, framealpha=0.2, loc="lower right")
plt.tight_layout()
plt.savefig("../trained_models/device_risk_models/fig3_feature_importance.png",
            dpi=150, bbox_inches="tight", facecolor="#0F172A")
print("📊 Figure 3 saved: fig3_feature_importance.png")

# ═══════════════════════════════════════════════════════════════════
# 11. FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  📈 DEVICE RISK — MODEL COMPARISON SUMMARY")
print("="*60)
print(f"  {'Model':<22} {'ROC-AUC':>8} {'F1':>7} {'Brier':>8}")
print("-"*60)
for name in names:
    marker = "  ← SELECTED" if name == "LightGBM" else ""
    print(f"  {name:<22} {trained[name]['roc_auc']:>8.4f} "
          f"{trained[name]['f1']:>7.4f} {trained[name]['brier']:>8.4f}{marker}")
print("="*60)
print("\n✅ All 3 figures + model saved.")
print("   fig1_performance_dashboard.png")
print("   fig2_probability_distribution.png")
print("   fig3_feature_importance.png")