import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Create Dataset
# -----------------------------

np.random.seed(42)
n = 200

study = np.random.uniform(0.5, 9, n)
attendance = np.random.uniform(40, 100, n)
sleep = np.random.uniform(4, 9, n)
assignment = np.random.uniform(30, 100, n)

# final score (simple formula + noise)
final = (
    5.2 * study +
    0.35 * attendance +
    1.8 * sleep +
    0.42 * assignment +
    np.random.normal(0, 4, n)
)

final = np.clip(final, 0, 100)

df = pd.DataFrame({
    "study_hours": np.round(study, 1),
    "attendance": np.round(attendance, 1),
    "sleep": np.round(sleep, 1),
    "assignment": np.round(assignment, 1),
    "final_score": np.round(final, 1)
})

print("\nFirst few rows:")
print(df.head())

# -----------------------------
# 2. Prepare Data
# -----------------------------

X = df[["study_hours", "attendance", "sleep", "assignment"]]
y = df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 3. Train Model
# -----------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 4. Evaluate
# -----------------------------

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\nModel Results:")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 3))

# -----------------------------
# 5. Graph (simple)
# -----------------------------

plt.scatter(y_test, pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()

# -----------------------------
# 6. Prediction Function
# -----------------------------

def predict():
    print("\nEnter your details:")

    study = float(input("Study hours: "))
    attendance = float(input("Attendance %: "))
    sleep = float(input("Sleep hours: "))
    assignment = float(input("Assignment score: "))

    data = pd.DataFrame([[study, attendance, sleep, assignment]],
                        columns=X.columns)

    data = scaler.transform(data)
    score = model.predict(data)[0]

    score = max(0, min(100, score))

    if score >= 90:
        grade = "A+"
    elif score >= 80:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 60:
        grade = "C"
    elif score >= 50:
        grade = "D"
    else:
        grade = "F"

    print("\nPredicted Score:", round(score, 1))
    print("Grade:", grade)


# -----------------------------
# Run
# -----------------------------

predict()
