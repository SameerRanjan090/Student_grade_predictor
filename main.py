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
