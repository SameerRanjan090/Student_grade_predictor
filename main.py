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
