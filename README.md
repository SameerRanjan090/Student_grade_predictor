
#  Student Grade Predictor

A Machine Learning project that predicts a student's final exam score based on their study habits and academic behaviour — built using **Linear Regression** in Python.

---

##  Problem Statement

Students often struggle to understand how their daily habits affect their final grades. This project builds a predictive model that takes four key inputs and estimates the final exam score, helping students make data-driven decisions about their study habits.

---

##  ML Approach

| Component | Detail |
|-----------|--------|
| Algorithm | Linear Regression |
| Library | scikit-learn |
| Features | Study hours, Attendance %, Sleep hours, Assignment score |
| Target | Final exam score (0–100) |
| Train/Test Split | 80% / 20% |
| Evaluation Metrics | MAE, RMSE, R² Score |

**Model Performance:**
- R² Score: **0.884** (88.4% variance explained)
- MAE: **3.90** (average error of ~4 marks)
- RMSE: **4.91**

---

##  Project Structure

```
student-grade-predictor/
│
├── student_grade_predictor.py   # Main ML script
├── grade_predictor_analysis.png # Auto-generated charts
└── README.md                    # This file
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/student-grade-predictor.git
cd student-grade-predictor
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib scikit-learn
```

### 3. Run the Project
```bash
python student_grade_predictor.py
```

---

##  How It Works

When you run the script, it will:

1. **Generate a dataset** of 200 synthetic students with realistic scores
2. **Analyse the data** — show statistics and feature correlations
3. **Train a Linear Regression model** on 80% of the data
4. **Evaluate the model** on the remaining 20%
5. **Save 3 charts** — Actual vs Predicted, Residuals, Feature Importance
6. **Ask you to enter your details** and predict your grade

### Example Interaction
```
  Daily study hours     (0.5 – 9.0) : 5
  Attendance percentage (40 – 100)  : 85
  Sleep hours per night (4 – 9)     : 7
  Assignment score      (0 – 100)   : 78

   Predicted Final Score : 91.3 / 100
   Predicted Grade       : A+
```

---

## 📊 Features Explained

| Feature | Description |
|---------|-------------|
| `study_hours` | Average hours spent studying per day |
| `attendance_pct` | Percentage of classes attended |
| `sleep_hours` | Average hours of sleep per night |
| `assignment_score` | Score obtained in assignments (out of 100) |

---

##  Sample Output Charts

The script automatically generates `grade_predictor_analysis.png` with:
- **Actual vs Predicted** scatter plot
- **Residual plot** to check model errors
- **Feature Importance** bar chart (based on scaled coefficients)

---

##  Dependencies

```
Python >= 3.8
pandas
numpy
matplotlib
scikit-learn
```

---

##  Author

**Sameer Ranjan**  
Course: Fundamentals of AIML  
Institution: VIT Bhopal  
Submission: BYOP — VITyarthi Platform
