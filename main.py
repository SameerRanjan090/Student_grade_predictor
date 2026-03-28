# ============================================================
#  Student Grade Predictor — BYOP ML Project
#  Predicts a student's final exam score using:
#    - Study hours per day
#    - Attendance percentage
#    - Hours of sleep per night
#    - Previous assignment score
#  Algorithm: Linear Regression (scikit-learn)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# 1. DATASET  (synthetic but realistic)
# ─────────────────────────────────────────────

np.random.seed(42)
n = 200

study_hours     = np.random.uniform(0.5, 9.0, n)
attendance      = np.random.uniform(40, 100, n)
sleep_hours     = np.random.uniform(4, 9, n)
assignment_score= np.random.uniform(30, 100, n)

# Final score is a weighted combination + small noise
final_score = (
    5.2  * study_hours +
    0.35 * attendance +
    1.8  * sleep_hours +
    0.42 * assignment_score +
    np.random.normal(0, 4, n)   # realistic noise
)
final_score = np.clip(final_score, 0, 100)   # keep scores in 0–100

df = pd.DataFrame({
    "study_hours"      : np.round(study_hours, 1),
    "attendance_pct"   : np.round(attendance, 1),
    "sleep_hours"      : np.round(sleep_hours, 1),
    "assignment_score" : np.round(assignment_score, 1),
    "final_score"      : np.round(final_score, 1),
})

print("=" * 55)
print("   STUDENT GRADE PREDICTOR — ML Project")
print("=" * 55)
print(f"\n Dataset shape : {df.shape[0]} students, {df.shape[1]} features")
print("\nFirst 5 rows:")
print(df.head().to_string(index=False))

# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────

print("\n\n Basic Statistics:")
print(df.describe().round(2).to_string())

# Correlation with final score
print("\n\n Correlation with Final Score:")
corr = df.corr()["final_score"].drop("final_score").sort_values(ascending=False)
for feat, val in corr.items():
    bar = "█" * int(abs(val) * 20)
    print(f"  {feat:<20} {val:+.3f}  {bar}")

# ─────────────────────────────────────────────
# 3. PREPARE DATA
# ─────────────────────────────────────────────

FEATURES = ["study_hours", "attendance_pct", "sleep_hours", "assignment_score"]
TARGET   = "final_score"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling (good practice)
scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"\n\n  Train size : {len(X_train)} | Test size : {len(X_test)}")

# ─────────────────────────────────────────────
# 4. TRAIN MODEL
# ─────────────────────────────────────────────

model = LinearRegression()
model.fit(X_train_s, y_train)

print("\n\n Model Trained — Coefficients:")
for feat, coef in zip(FEATURES, model.coef_):
    print(f"  {feat:<22} {coef:+.4f}")
print(f"  {'Intercept':<22} {model.intercept_:+.4f}")

# ─────────────────────────────────────────────
# 5. EVALUATE MODEL
# ─────────────────────────────────────────────

y_pred = model.predict(X_test_s)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("\n\n Model Evaluation on Test Set:")
print(f"  Mean Absolute Error  (MAE)  : {mae:.2f}")
print(f"  Root Mean Sq. Error  (RMSE) : {rmse:.2f}")
print(f"  R² Score                    : {r2:.4f}  ({r2*100:.1f}% variance explained)")

# ─────────────────────────────────────────────
# 6. VISUALISATIONS  (saved as PNG)
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Student Grade Predictor — Analysis", fontsize=14, fontweight="bold")

# Plot 1 — Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.6, color="steelblue", edgecolors="white", s=60)
mn, mx = y_test.min(), y_test.max()
axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect prediction")
axes[0].set_xlabel("Actual Score")
axes[0].set_ylabel("Predicted Score")
axes[0].set_title("Actual vs Predicted")
axes[0].legend()

# Plot 2 — Residuals
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.6, color="coral", edgecolors="white", s=60)
axes[1].axhline(0, color="black", linewidth=1.2, linestyle="--")
axes[1].set_xlabel("Predicted Score")
axes[1].set_ylabel("Residual (Actual − Predicted)")
axes[1].set_title("Residual Plot")

# Plot 3 — Feature importance (abs coefficient values)
importance = np.abs(model.coef_)
feat_labels = ["Study Hrs", "Attendance", "Sleep Hrs", "Assignment"]
colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
bars = axes[2].bar(feat_labels, importance, color=colors, edgecolor="white")
axes[2].set_title("Feature Importance\n(|Coefficient| after scaling)")
axes[2].set_ylabel("|Coefficient|")
for bar, val in zip(bars, importance):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("grade_predictor_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n   Chart saved → grade_predictor_analysis.png")

# ─────────────────────────────────────────────
# 7. INTERACTIVE PREDICTOR
# ─────────────────────────────────────────────

def predict_grade(study, attendance, sleep, assignment):
    """Predict the final score for a single student."""
    sample = pd.DataFrame([[study, attendance, sleep, assignment]], columns=FEATURES)
    sample_s = scaler.transform(sample)
    score = model.predict(sample_s)[0]
    score = float(np.clip(score, 0, 100))

    if score >= 90:   grade = "A+"
    elif score >= 80: grade = "A"
    elif score >= 70: grade = "B"
    elif score >= 60: grade = "C"
    elif score >= 50: grade = "D"
    else:             grade = "F"

    return score, grade


def run_predictor():
    print("\n\n" + "=" * 55)
    print("    PREDICT YOUR GRADE")
    print("=" * 55)

    def get_float(prompt, lo, hi):
        while True:
            try:
                val = float(input(prompt))
                if lo <= val <= hi:
                    return val
                print(f"     Please enter a value between {lo} and {hi}.")
            except ValueError:
                print("     Invalid input. Enter a number.")

    study      = get_float("  Daily study hours     (0.5 – 9.0) : ", 0.5, 9.0)
    attendance = get_float("  Attendance percentage (40 – 100)  : ", 40, 100)
    sleep      = get_float("  Sleep hours per night (4 – 9)     : ", 4, 9)
    assignment = get_float("  Assignment score      (0 – 100)   : ", 0, 100)

    score, grade = predict_grade(study, attendance, sleep, assignment)

    print("\n" + "-" * 45)
    print(f"  Predicted Final Score : {score:.1f} / 100")
    print(f"  Predicted Grade       : {grade}")
    print("-" * 45)

    # Tips
    print("\n   Tips to improve your score:")
    if study < 4:
        print("   • Try studying at least 4 hours daily.")
    if attendance < 75:
        print("   • Aim for 75%+ attendance — it matters!")
    if sleep < 6:
        print("   • Get at least 6–7 hours of sleep.")
    if assignment < 60:
        print("   • Focus on assignment scores — they correlate strongly.")
