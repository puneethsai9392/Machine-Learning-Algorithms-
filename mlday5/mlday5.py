import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection    import train_test_split, cross_val_score
from sklearn.linear_model       import LinearRegression
from sklearn.preprocessing      import StandardScaler, PolynomialFeatures
from sklearn.pipeline           import Pipeline
from sklearn.metrics            import mean_squared_error, mean_absolute_error, r2_score

# ---------------- generate random data ----------------
n_sample = 300
stud_hour  = np.random.randint(3, 10, n_sample)
past_marks = np.random.randint(1, 100, n_sample)
sleep_h    = np.random.randint(4, 10, n_sample)

noise = np.random.normal(0, 5, n_sample)

marks = (
    0.4 * stud_hour**2
    + 0.15 * past_marks
    + 0.8 * sleep_h
    + 0.3 * stud_hour * sleep_h
    + noise
    + 20
)
marks = np.clip(marks, 0, 100)

df = pd.DataFrame({
    'study_hour': stud_hour,
    'past_marks': past_marks,
    'sleep_h'   : sleep_h,
    'marks'     : marks
})

print("-" * 50)
print("First 5 rows of the dataset")
print("-" * 50)
print(df.head())

print("-" * 50)
print("Explore the dataset")
print("-" * 50)
print(df.info())

print("-" * 50)
print("Number of null values")
print("-" * 50)
print(df.isnull().sum())

print("-" * 50)
print("Dataset description")
print("-" * 50)
print(df.describe())

print("-" * 50)
print("Correlation matrix")
print("-" * 50)
print(df.corr()['marks'].sort_values(ascending=False))

print(f"\nThe shape of the dataset is: {df.shape}")

X = df.drop(columns=['marks'])
y = df['marks']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- degree selection via CV ----------------
results = {}
for degree in range(1, 11):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('lr', LinearRegression())
    ])
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    results[degree] = scores.mean()

for degree, score in results.items():
    print(f"Degree = {degree}, Mean R² = {score:.4f}")

best_degree = max(results, key=results.get)
print(f"\nBest Degree: {best_degree}")
print(f"Best Mean R²: {results[best_degree]:.4f}")

# ---------------- final model with best degree ----------------
model = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=best_degree, include_bias=False)),
    ('lr', LinearRegression())
])

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)


def evalute(y_true, y_pred, label):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    print('-' * 60)
    print('Model evalution metrics -->', label)
    print('-' * 60)
    print('mean_absolute_error     :', mae)
    print('mean_squared_error      :', mse)
    print('root_mean_squared_error :', rmse)
    print('r2_score                :', r2)
    return mae, mse, rmse, r2


test_mae, test_mse, test_rmse, test_r2   = evalute(y_test, y_test_pred, 'Test_data')
train_mae, train_mse, train_rmse, train_r2 = evalute(y_train, y_train_pred, 'Train_data')

print()
print('-' * 60)
poly_feature_names = model.named_steps['poly'].get_feature_names_out(X.columns)
for fet, coef_ in zip(poly_feature_names, model.named_steps['lr'].coef_):
    print(f'{fet:<25}: {coef_:.4f}')
print(f"the model intercept is       :{model.named_steps['lr'].intercept_:.2f}")
print('-' * 60)

test_residuals  = y_test - y_test_pred
train_residuals = y_train - y_train_pred

print('Test residuals  mean :', test_residuals.mean())
print('Test residuals  std  :', test_residuals.std())
print('Train residuals mean :', train_residuals.mean())
print('Train residuals std  :', train_residuals.std())


plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    f'Polynomial Regression — Marks Prediction (Best Degree = {best_degree})\nR2 Score (Test): {test_r2:.4f}',
    fontsize=18,
    fontweight='bold'
)

# 1. Degree vs Mean R2 (CV curve)
ax1 = axes[0, 0]
degrees = list(results.keys())
scores_ = list(results.values())
ax1.plot(degrees, scores_, marker='o', color='purple', lw=2)
ax1.axvline(best_degree, color='red', linestyle='--', alpha=0.7, label=f'Best = {best_degree}')
ax1.set_xlabel('Polynomial Degree', fontsize=10, fontweight='bold')
ax1.set_ylabel('Mean CV R²', fontsize=10, fontweight='bold')
ax1.set_title('1. Degree Selection (CV)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.4)

# 2. Actual vs Predicted (Train)
ax2 = axes[0, 1]
ax2.scatter(y_train, y_train_pred, s=70, alpha=0.7, color='orange', edgecolor='white', label='Training Data')
y_min, y_max = y.min(), y.max()
ax2.plot([y_min, y_max], [y_min, y_max], alpha=0.6, lw=1.6, color='red', label='Ideal Fit')
ax2.set_xlabel('Actual Values', fontsize=10, fontweight='bold')
ax2.set_ylabel('Predicted Values', fontsize=10, fontweight='bold')
ax2.set_title('2. Actual Vs Predicted (Train)', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.4)

# 3. Actual vs Predicted (Test)
ax3 = axes[0, 2]
ax3.scatter(y_test, y_test_pred, s=70, alpha=0.7, color='dodgerblue', edgecolor='white', label='Test Data')
ax3.plot([y_min, y_max], [y_min, y_max], alpha=0.6, lw=1.6, color='red', label='Ideal Fit')
ax3.set_xlabel('Actual Values', fontsize=10, fontweight='bold')
ax3.set_ylabel('Predicted Values', fontsize=10, fontweight='bold')
ax3.set_title('3. Actual Vs Predicted (Test)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.4)


ax4 = axes[1, 0]
ax4.scatter(y_test_pred, test_residuals, s=70, alpha=0.7, color='green', edgecolors='white', label='Residuals')
ax4.axhline(y=0, lw=2, linestyle='--', color='red', alpha=0.7, label='Zero Errors')
ax4.set_xlabel('Predicted Values', fontsize=10, fontweight='bold')
ax4.set_ylabel('Residuals', fontsize=10, fontweight='bold')
ax4.set_title('4. Residuals Vs Predicted (Test)', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.5)


ax5 = axes[1, 1]
ax5.hist(train_residuals, bins=25, edgecolor='white', color='violet', label='Train Residuals')
ax5.set_xlabel('Residuals', fontsize=10, fontweight='bold')
ax5.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax5.set_title('5. Residuals Distribution (Train)', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.5)

# 6. Q-Q Plot of Test Residuals
ax6 = axes[1, 2]
stats.probplot(test_residuals, dist="norm", plot=ax6)
ax6.get_lines()[0].set_markerfacecolor('teal')
ax6.get_lines()[0].set_markeredgecolor('white')
ax6.get_lines()[0].set_markersize(6)
ax6.get_lines()[1].set_color('red')
ax6.set_title('6. Q-Q Plot Of Test Residuals', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.5)

plt.tight_layout(rect=[0, 0.02, 1, 0.94])
plt.savefig('Polynomial Regression Data SET.png', dpi=300, bbox_inches='tight')
print('Polynomial regression dashboard saved!')
plt.show()