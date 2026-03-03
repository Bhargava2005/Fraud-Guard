import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, brier_score_loss, f1_score,
    precision_score, recall_score, precision_recall_curve,
    average_precision_score
)

# ═══════════════════════════════════════════════════════════════
#  STYLE CONFIG
# ═══════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor"  : "#0F1117",
    "axes.facecolor"    : "#1A1D27",
    "axes.edgecolor"    : "#2E3250",
    "axes.labelcolor"   : "#C8CEED",
    "axes.titlecolor"   : "#FFFFFF",
    "xtick.color"       : "#8890B5",
    "ytick.color"       : "#8890B5",
    "text.color"        : "#C8CEED",
    "grid.color"        : "#2E3250",
    "grid.linewidth"    : 0.6,
    "font.family"       : "DejaVu Sans",
    "axes.titlesize"    : 13,
    "axes.labelsize"    : 11,
    "legend.facecolor"  : "#1A1D27",
    "legend.edgecolor"  : "#2E3250",
})

COLORS = {
    "lgbm"    : "#00D4FF",
    "xgb"     : "#FF6B6B",
    "rf"      : "#51CF66",
    "gb"      : "#FFD43B",
    "lr"      : "#CC5DE8",
    "positive": "#FF4757",
    "negative": "#2ED573",
    "accent"  : "#00D4FF",
}

MODEL_COLORS = {
    "LightGBM"           : COLORS["lgbm"],
    "XGBoost"            : COLORS["xgb"],
    "Random Forest"      : COLORS["rf"],
    "Gradient Boosting"  : COLORS["gb"],
    "Logistic Regression": COLORS["lr"],
}

# ═══════════════════════════════════════════════════════════════
#  1. LOAD & PREPARE DATA
# ═══════════════════════════════════════════════════════════════
print("📦 Loading data...")
data = pd.read_csv("../assets/customer_risk_data.csv")
y    = data["customer_risk_label"]
X    = data.drop(["pin_code", "customer_risk_label"], axis=1)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, stratify=y, test_size=0.2, random_state=42
)

# ═══════════════════════════════════════════════════════════════
#  2. DEFINE & TRAIN MODELS
# ═══════════════════════════════════════════════════════════════
base_models = {
    "LightGBM"           : LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight="balanced", random_state=42, verbose=-1),
    "XGBoost"            : XGBClassifier(n_estimators=300, learning_rate=0.05, scale_pos_weight=(y==0).sum()/(y==1).sum(), random_state=42, eval_metric="logloss", verbosity=0),
    "Random Forest"      : RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting"  : GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
}

print("🔧 Training and calibrating all models...")
results = {}
for name, base in base_models.items():
    print(f"   ▸ {name}...")
    cal = CalibratedClassifierCV(base, method="isotonic", cv=5)
    cal.fit(X_train, y_train)
    y_pred  = cal.predict(X_test)
    y_proba = cal.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    fraction_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10)
    results[name] = {
        "model"       : cal,
        "y_pred"      : y_pred,
        "y_proba"     : y_proba,
        "roc_auc"     : roc_auc_score(y_test, y_proba),
        "f1"          : f1_score(y_test, y_pred),
        "precision"   : precision_score(y_test, y_pred),
        "recall"      : recall_score(y_test, y_pred),
        "brier"       : brier_score_loss(y_test, y_proba),
        "fpr"         : fpr,
        "tpr"         : tpr,
        "prec_curve"  : prec,
        "rec_curve"   : rec,
        "avg_prec"    : average_precision_score(y_test, y_proba),
        "frac_pos"    : fraction_pos,
        "mean_pred"   : mean_pred,
    }

lgbm_res = results["LightGBM"]
lgbm_cm  = confusion_matrix(y_test, lgbm_res["y_pred"])

# Cross-validation scores for LightGBM
print("   ▸ Running cross-validation for LightGBM...")
lgbm_cv = cross_val_score(
    CalibratedClassifierCV(base_models["LightGBM"], method="isotonic", cv=3),
    X_scaled, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc", n_jobs=-1
)

# Feature importance from raw LightGBM
raw_lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight="balanced", random_state=42, verbose=-1)
raw_lgbm.fit(X_train, y_train)
feat_imp = pd.Series(raw_lgbm.feature_importances_, index=X.columns).sort_values(ascending=True)

print("✅ All models trained!\n")

# ═══════════════════════════════════════════════════════════════
#  3. FIGURE 1 — MODEL COMPARISON DASHBOARD (6 plots)
# ═══════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(22, 16), facecolor="#0F1117")
fig1.suptitle("CUSTOMER FRAUD RISK — MODEL COMPARISON DASHBOARD",
              fontsize=18, fontweight="bold", color="#FFFFFF", y=0.98)

gs1 = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.42, wspace=0.35,
                        left=0.06, right=0.97, top=0.93, bottom=0.07)

# ── Plot 1: ROC Curves ──────────────────────────────────────────
ax1 = fig1.add_subplot(gs1[0, 0])
ax1.set_title("ROC Curves — All Models", pad=10)
ax1.plot([0,1],[0,1], ":", color="#555", linewidth=1.2, label="Random (AUC=0.50)")
for name, res in results.items():
    lw = 2.8 if name == "LightGBM" else 1.6
    ls = "-"  if name == "LightGBM" else "--"
    ax1.plot(res["fpr"], res["tpr"], color=MODEL_COLORS[name],
             linewidth=lw, linestyle=ls,
             label=f"{name} ({res['roc_auc']:.3f})")
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.legend(fontsize=7.5, loc="lower right")
ax1.grid(True, alpha=0.3)
ax1.set_facecolor("#1A1D27")

# ── Plot 2: Precision-Recall Curves ────────────────────────────
ax2 = fig1.add_subplot(gs1[0, 1])
ax2.set_title("Precision–Recall Curves", pad=10)
for name, res in results.items():
    lw = 2.8 if name == "LightGBM" else 1.6
    ls = "-"  if name == "LightGBM" else "--"
    ax2.plot(res["rec_curve"], res["prec_curve"], color=MODEL_COLORS[name],
             linewidth=lw, linestyle=ls,
             label=f"{name} (AP={res['avg_prec']:.3f})")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.legend(fontsize=7.5, loc="upper right")
ax2.grid(True, alpha=0.3)
ax2.set_facecolor("#1A1D27")

# ── Plot 3: Metrics Bar Chart ───────────────────────────────────
ax3 = fig1.add_subplot(gs1[0, 2])
ax3.set_title("Model Metrics Comparison", pad=10)
metrics_names  = ["ROC-AUC", "F1", "Precision", "Recall"]
metric_keys    = ["roc_auc", "f1", "precision", "recall"]
x_pos          = np.arange(len(metrics_names))
bar_width      = 0.15
offsets        = np.linspace(-0.3, 0.3, len(results))
for idx, (name, res) in enumerate(results.items()):
    vals = [res[k] for k in metric_keys]
    bars = ax3.bar(x_pos + offsets[idx], vals, bar_width,
                   color=MODEL_COLORS[name], alpha=0.85, label=name)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(metrics_names, fontsize=9)
ax3.set_ylim(0.5, 1.05)
ax3.set_ylabel("Score")
ax3.legend(fontsize=7, loc="lower right")
ax3.grid(True, alpha=0.3, axis="y")
ax3.set_facecolor("#1A1D27")

# ── Plot 4: Confusion Matrix (LightGBM) ────────────────────────
ax4 = fig1.add_subplot(gs1[1, 0])
ax4.set_title("Confusion Matrix — LightGBM", pad=10)
cmap_cm = LinearSegmentedColormap.from_list("cm", ["#1A1D27", "#00D4FF"])
im = ax4.imshow(lgbm_cm, interpolation="nearest", cmap=cmap_cm)
ax4.set_xticks([0,1]); ax4.set_yticks([0,1])
ax4.set_xticklabels(["Pred: Low Risk", "Pred: High Risk"], fontsize=9)
ax4.set_yticklabels(["Actual: Low Risk", "Actual: High Risk"], fontsize=9)
ax4.set_xlabel("Predicted Label"); ax4.set_ylabel("Actual Label")
thresh = lgbm_cm.max() / 2
for i in range(2):
    for j in range(2):
        val   = lgbm_cm[i, j]
        color = "white" if val < thresh else "#0F1117"
        ax4.text(j, i, f"{val:,}", ha="center", va="center",
                 fontsize=14, fontweight="bold", color=color)
labels = ["TN", "FP", "FN", "TP"]
for idx, (i, j) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
    ax4.text(j, i+0.32, labels[idx], ha="center", va="center",
             fontsize=8, color="#8890B5")
fig1.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

# ── Plot 5: Feature Importance ──────────────────────────────────
ax5 = fig1.add_subplot(gs1[1, 1])
ax5.set_title("Feature Importance — LightGBM", pad=10)
colors_fi = [COLORS["accent"] if v == feat_imp.max() else "#3A4080" for v in feat_imp.values]
bars = ax5.barh(feat_imp.index, feat_imp.values, color=colors_fi, edgecolor="#2E3250", height=0.6)
for bar, val in zip(bars, feat_imp.values):
    ax5.text(val + feat_imp.max()*0.01, bar.get_y() + bar.get_height()/2,
             f"{val:,}", va="center", fontsize=8, color="#C8CEED")
ax5.set_xlabel("Importance Score")
ax5.grid(True, alpha=0.3, axis="x")
ax5.set_facecolor("#1A1D27")

# ── Plot 6: Probability Distribution ───────────────────────────
ax6 = fig1.add_subplot(gs1[1, 2])
ax6.set_title("Risk Probability Distribution — LightGBM", pad=10)
proba_0 = lgbm_res["y_proba"][y_test == 0]
proba_1 = lgbm_res["y_proba"][y_test == 1]
ax6.hist(proba_0, bins=40, alpha=0.7, color=COLORS["negative"],
         label="Actual Low Risk (0)", density=True)
ax6.hist(proba_1, bins=40, alpha=0.7, color=COLORS["positive"],
         label="Actual High Risk (1)", density=True)
ax6.axvline(0.5, color="white", linewidth=1.5, linestyle="--", label="Threshold = 0.5")
ax6.set_xlabel("Predicted Risk Probability")
ax6.set_ylabel("Density")
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)
ax6.set_facecolor("#1A1D27")

plt.savefig("../trained_models/customer_risk_models/fig1_model_comparison.png",
            dpi=150, bbox_inches="tight", facecolor="#0F1117")
print("💾 Saved: fig1_model_comparison.png")

# ═══════════════════════════════════════════════════════════════
#  4. FIGURE 2 — LIGHTGBM DEEP DIVE (4 plots)
# ═══════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(20, 12), facecolor="#0F1117")
fig2.suptitle("LightGBM — DEEP PERFORMANCE ANALYSIS",
              fontsize=18, fontweight="bold", color="#FFFFFF", y=0.98)

gs2 = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.40, wspace=0.32,
                        left=0.07, right=0.97, top=0.92, bottom=0.08)

# ── Plot 7: Cross-Validation Scores ────────────────────────────
ax7 = fig2.add_subplot(gs2[0, 0])
ax7.set_title("5-Fold Cross-Validation — ROC-AUC", pad=10)
fold_nums = [f"Fold {i+1}" for i in range(len(lgbm_cv))]
bar_colors = [COLORS["accent"] if v == lgbm_cv.max() else "#3A4080" for v in lgbm_cv]
bars7 = ax7.bar(fold_nums, lgbm_cv, color=bar_colors, edgecolor="#2E3250", width=0.5)
ax7.axhline(lgbm_cv.mean(), color="#FFD43B", linewidth=2, linestyle="--",
            label=f"Mean = {lgbm_cv.mean():.4f}")
ax7.axhline(lgbm_cv.mean() - lgbm_cv.std(), color="#FF6B6B", linewidth=1,
            linestyle=":", label=f"±1 Std = {lgbm_cv.std():.4f}")
ax7.axhline(lgbm_cv.mean() + lgbm_cv.std(), color="#FF6B6B", linewidth=1, linestyle=":")
for bar, val in zip(bars7, lgbm_cv):
    ax7.text(bar.get_x() + bar.get_width()/2, val + 0.001,
             f"{val:.4f}", ha="center", fontsize=9, color="white", fontweight="bold")
ax7.set_ylim(lgbm_cv.min() - 0.05, 1.02)
ax7.set_ylabel("ROC-AUC Score")
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3, axis="y")
ax7.set_facecolor("#1A1D27")

# ── Plot 8: Calibration Curve ───────────────────────────────────
ax8 = fig2.add_subplot(gs2[0, 1])
ax8.set_title("Probability Calibration — All Models", pad=10)
ax8.plot([0,1],[0,1], "w--", linewidth=1.5, label="Perfect Calibration")
for name, res in results.items():
    lw = 2.8 if name == "LightGBM" else 1.6
    ax8.plot(res["mean_pred"], res["frac_pos"], "o-",
             color=MODEL_COLORS[name], linewidth=lw, markersize=5,
             label=name)
ax8.set_xlabel("Mean Predicted Probability")
ax8.set_ylabel("Fraction of Positives")
ax8.legend(fontsize=8)
ax8.grid(True, alpha=0.3)
ax8.set_facecolor("#1A1D27")

# ── Plot 9: Metrics Summary Scorecard ──────────────────────────
ax9 = fig2.add_subplot(gs2[1, 0])
ax9.set_title("LightGBM — Metrics Scorecard", pad=10)
ax9.axis("off")
scorecard_metrics = {
    "ROC-AUC Score"   : f"{lgbm_res['roc_auc']:.4f}",
    "F1 Score"        : f"{lgbm_res['f1']:.4f}",
    "Precision"       : f"{lgbm_res['precision']:.4f}",
    "Recall"          : f"{lgbm_res['recall']:.4f}",
    "Brier Score"     : f"{lgbm_res['brier']:.4f}",
    "CV Mean AUC"     : f"{lgbm_cv.mean():.4f}",
    "CV Std AUC"      : f"{lgbm_cv.std():.4f}",
    "True Positives"  : f"{lgbm_cm[1,1]:,}",
    "True Negatives"  : f"{lgbm_cm[0,0]:,}",
    "False Positives" : f"{lgbm_cm[0,1]:,}",
    "False Negatives" : f"{lgbm_cm[1,0]:,}",
}
y_pos = 0.96
for i, (metric, value) in enumerate(scorecard_metrics.items()):
    bg_color = "#1E2235" if i % 2 == 0 else "#232840"
    ax9.add_patch(mpatches.FancyBboxPatch(
        (0.01, y_pos - 0.085), 0.98, 0.082,
        boxstyle="round,pad=0.005", facecolor=bg_color,
        edgecolor="#2E3250", transform=ax9.transAxes
    ))
    color = COLORS["accent"] if "AUC" in metric or "F1" in metric else "#C8CEED"
    ax9.text(0.08, y_pos - 0.042, metric, transform=ax9.transAxes,
             fontsize=9.5, color="#8890B5", va="center")
    ax9.text(0.92, y_pos - 0.042, value, transform=ax9.transAxes,
             fontsize=10, color=color, va="center", ha="right", fontweight="bold")
    y_pos -= 0.088
ax9.set_facecolor("#1A1D27")

# ── Plot 10: Brier Score Comparison ─────────────────────────────
ax10 = fig2.add_subplot(gs2[1, 1])
ax10.set_title("Brier Score Comparison\n(Lower = Better Probability Calibration)", pad=10)
brier_names  = list(results.keys())
brier_values = [results[n]["brier"] for n in brier_names]
bar_colors10 = [COLORS["accent"] if n == "LightGBM" else MODEL_COLORS[n] for n in brier_names]
bars10 = ax10.barh(brier_names, brier_values, color=bar_colors10,
                   edgecolor="#2E3250", height=0.5)
for bar, val in zip(bars10, brier_values):
    ax10.text(val + 0.001, bar.get_y() + bar.get_height()/2,
              f"{val:.4f}", va="center", fontsize=9, color="white")
ax10.set_xlabel("Brier Score")
ax10.grid(True, alpha=0.3, axis="x")
ax10.set_facecolor("#1A1D27")
best_brier = min(brier_values)
ax10.axvline(best_brier, color="#FFD43B", linestyle="--", linewidth=1.5,
             label=f"Best = {best_brier:.4f}")
ax10.legend(fontsize=9)

plt.savefig("../trained_models/customer_risk_models/fig2_lgbm_deep_dive.png",
            dpi=150, bbox_inches="tight", facecolor="#0F1117")
print("💾 Saved: fig2_lgbm_deep_dive.png")

# ═══════════════════════════════════════════════════════════════
#  5. SAVE FINAL MODEL
# ═══════════════════════════════════════════════════════════════
joblib.dump(lgbm_res["model"],  "../trained_models/customer_risk_models/customer_risk_model.pkl")
joblib.dump(scaler,             "../trained_models/customer_risk_models/customer_risk_scaler.pkl")
joblib.dump(X.columns.tolist(), "../trained_models/customer_risk_models/customer_risk_features.pkl")

# ═══════════════════════════════════════════════════════════════
#  6. FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  📊 FINAL MODEL PERFORMANCE SUMMARY")
print("=" * 60)
print(f"  {'Model':<22} {'AUC':>7} {'F1':>7} {'Brier':>7}")
print("-" * 60)
for name, res in results.items():
    marker = " ← SELECTED" if name == "LightGBM" else ""
    print(f"  {name:<22} {res['roc_auc']:>7.4f} {res['f1']:>7.4f} {res['brier']:>7.4f}{marker}")
print("=" * 60)
print(f"\n  ✅ Best Model   : LightGBM")
print(f"  📈 ROC-AUC      : {lgbm_res['roc_auc']:.4f}")
print(f"  🎯 F1 Score     : {lgbm_res['f1']:.4f}")
print(f"  🔁 CV Mean AUC  : {lgbm_cv.mean():.4f} ± {lgbm_cv.std():.4f}")
print(f"\n  📁 Saved Figures:")
print(f"     → fig1_model_comparison.png")
print(f"     → fig2_lgbm_deep_dive.png")
print("=" * 60)

plt.show()