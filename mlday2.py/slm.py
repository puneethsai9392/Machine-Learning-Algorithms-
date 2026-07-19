import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===========================
# Load Dataset
# ===========================
df = pd.read_csv(r"C:\Users\Hp\Documents\simple_linear_day1\env\mlday2\mlday2.py\slm.csv")   # Replace with your CSV file name

# Select independent and dependent variables
X = df[["kilometers"]]
y = df["charge"]

# ===========================
# Explore Dataset
# ===========================
print("=" * 50)
print("First 5 rows of the dataset")
print("=" * 50)
print(df.head())

print("=" * 50)
print("Explore the dataset")
print("=" * 50)
print(df.info())

print("=" * 50)
print("Number of null values")
print("=" * 50)
print(df.isnull().sum())

print("=" * 50)
print("Dataset description")
print("=" * 50)
print(df.describe())

print("=" * 50)
print("Correlation matrix")
print("=" * 50)
print(df.corr(numeric_only=True))

print(f"\nThe shape of the dataset is: {df.shape}")

# ===========================
# Train-Test Split
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===========================
# Train Model
# ===========================
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# ===========================
# Evaluation Function
# ===========================
def evaluate(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print("\n" + "=" * 50)
    print(f"Model Evaluation Metrics for {name}")
    print("=" * 50)
    print(f"Mean Squared Error (MSE): {mse:.3f}")
    print(f"Mean Absolute Error (MAE): {mae:.3f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
    print(f"R-squared (R2): {r2:.3f}")

evaluate(y_test, y_test_pred, "Test Set")
evaluate(y_train, y_train_pred, "Train Set")

# ===========================
# Model Parameters
# ===========================
print("\n" + "=" * 50)
print("Model Parameters")
print("=" * 50)
print(f"Model Coefficient: {model.coef_[0]:.6f}")
print(f"Model Intercept: {model.intercept_:.6f}")

# ===========================
# Prediction
# ===========================
km = float(input("Enter kilometers: "))
prediction = model.predict([[km]])
print(f"Predicted Charge: {prediction[0]:.2f}")

# ===========================
# Visualization
# ===========================
plt.style.use("seaborn-v0_8-whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

fig.suptitle(
    f"Simple Linear Regression Model: Kilometers vs Charge\nR2 Score: {r2_score(y_test, y_test_pred):.2f}",
    fontsize=18,
    fontweight="bold"
)

# Plot 1: Regression Line
axes[0,0].scatter(X_test, y_test, color="orange", edgecolors="black", s=80, alpha=0.7, label="Testing Data")
X_line = np.linspace(X.min().values[0], X.max().values[0], 100).reshape(-1,1)
axes[0,0].plot(X_line, model.predict(X_line), color="red", linewidth=2, label="Regression Line")
axes[0,0].set_title("Testing Data and Regression Line")
axes[0,0].set_xlabel("Kilometers")
axes[0,0].set_ylabel("Charge")
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.5)

# Plot 2: Actual vs Predicted
axes[0,1].scatter(y_test, y_test_pred, color="green", edgecolors="black", s=80, alpha=0.7,label="Testing Data")
axes[0,1].plot([y.min(), y.max()], [y.min(), y.max()],color="red",linewidth=2, label="Perfect Prediction")
axes[0,1].set_title("Actual vs Predicted Values")
axes[0,1].set_xlabel("Actual Values")
axes[0,1].set_ylabel("Predicted Values")
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.5)

# Plot 3: Residual Plot
residuals = y_test - y_test_pred
axes[1,0].scatter(y_test, residuals, color="violet", edgecolors="black", s=80, alpha=0.7,label="Residuals")
axes[1,0].axhline(y=0, color="red", linewidth=2,label="Zero Error Line")
axes[1,0].set_title("Residuals Plot")
axes[1,0].set_xlabel("Actual Values")
axes[1,0].set_ylabel("Residuals")
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.5)

# Plot 4: Residual Distribution
axes[1,1].hist(residuals, bins=20, color="lightblue", edgecolor="black", alpha=0.7, label="Residuals Distribution")
axes[1,1].set_title("Residuals Distribution")
axes[1,1].set_xlabel("Residuals")
axes[1,1].set_ylabel("Frequency")
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.savefig("simple_linear_regression_plots.png", dpi=300)
plt.show()

print("\nPlots saved as 'simple_linear_regression_plots.png'")