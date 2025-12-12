import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import joblib

# Load dataset
df = pd.read_csv("Datasets/garments_worker_productivity.csv")


# Convert numeric-looking columns to numeric type
df["smv"] = pd.to_numeric(df["smv"], errors="coerce")
df["no_of_workers"] = pd.to_numeric(df["no_of_workers"], errors="coerce")

# Fill missing numeric values
df["smv"].fillna(df["smv"].median(), inplace=True)
df["no_of_workers"].fillna(df["no_of_workers"].median(), inplace=True)


# ------------------------------
#  FEATURE ENGINEERING
# ------------------------------

# 1. Productivity gap (train only)
df["productivity_gap"] = df["targeted_productivity"] - df["actual_productivity"]

# 2. Over-time per worker
df["overtime_per_worker"] = df["over_time"] / (df["no_of_workers"] + 1)

# 3. Idle ratio
df["idle_ratio"] = df["idle_time"] / (df["idle_men"] + 1)

# 4. Saturday indicator
df["is_saturday"] = (df["day"] == "Saturday").astype(int)

# 5. Team stress = style changes per team
df["team_stress"] = df["no_of_style_change"] / (df["team"] + 1)

# 6. SMV per worker
df["smv_per_worker"] = df["smv"] / (df["no_of_workers"] + 1)

# ------------------------------
# SELECT FEATURES
# ------------------------------

target = "actual_productivity"

feature_columns = [
    'quarter', 'department', 'day', 'team',
    'targeted_productivity', 'smv', 'wip', 'over_time',
    'incentive', 'idle_time', 'idle_men', 'no_of_style_change',
    'no_of_workers',

    # engineered
    'productivity_gap',
    'overtime_per_worker',
    'idle_ratio',
    'is_saturday',
    'team_stress',
    'smv_per_worker'
]

X = df[feature_columns].copy()
y = df[target].copy()

# Handle missing numeric values
numeric_cols = X.select_dtypes(include=['int64','float64']).columns
numeric_imputer = SimpleImputer(strategy="median")
X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])

# Handle categorical
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
#    TRAIN XGBOOST REGRESSOR
# ------------------------------

model = XGBRegressor(
    n_estimators=700,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Save
joblib.dump(model, "xgb_feature_engineered_model.pkl")

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("====== XGBOOST + FEATURE ENGINEERING RESULTS ======")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R-squared: {r2:.4f}")
