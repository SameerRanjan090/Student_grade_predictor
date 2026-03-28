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
