import warnings 
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import train_test_split
from sklearn.linear_model     import LinearRegression
from sklearn.preprocessing    import StandardScaler,PolynomialFeatures
from sklearn.metrics          import mean_squared_error,r2_score,mean_absolute_error
from sklearn.pipeline         import Pipeline


n_sample=200
skills=np.random.randint(4,15,n_sample)
experience=np.random.uniform(0,10,n_sample)
graduation=np.random.randint(12,18,n_sample)
noise=np.random.randint(-4000,4000,n_sample)
salary=(
    skills*1000+
    experience*1500+
    graduation*500+1000+
    noise
)
df=pd.DataFrame({'skills':skills,
                'experience':experience ,
                'graduation':graduation,
                'salary':salary
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
print(df.corr()['salary'].sort_values(ascending=False))

print(f"\nThe shape of the dataset is: {df.shape}")

X=df.drop(columns='salary')

y=df['salary']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=Pipeline([('scaler',StandardScaler()),
                ('linear',LinearRegression())])
model.fit(X_train,y_train)
y_pred_test=model.predict(X_test)
y_pred_train=model.predict(X_train)

def evaluate(y_true,y_pred,label):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print("\n" + "=" * 50)
    print(f"Model Evaluation Metrics for {label}")
    print("=" * 50)
    print(f"Mean Squared Error (MSE): {mse:.3f}")
    print(f"Mean Absolute Error (MAE): {mae:.3f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
    print(f"R-squared (R2): {r2:.3f}")

evaluate(y_test, y_pred_test, "Test Set")
evaluate(y_train, y_pred_train, "Train Set")

print('-'*60)
print(f"\nTrain Score : {model.score(X_train, y_train):.4f}")
print(f"Test Score  : {model.score(X_test, y_test):.4f}")


print("\n" + "=" * 50)
print("Model Parameters")
print("=" * 50)

for fet,coef in zip(X.columns,model.named_steps['linear'].coef_):
    print(f'{fet:<13}:{coef:.3f}')

print(f"Model Intercept: {model.named_steps['linear'].intercept_:.6f}")

residuals_train = y_train - y_pred_train
residuals_test = y_test - y_pred_test
print(f'the mean of train residuals :{residuals_train.mean()}')
print(f'the train residuals std :{residuals_train.std()}')
print(f'the mean of test residuals  :{residuals_test.mean()}')
print(f'the test residuals std  :{residuals_test.std()}')


plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(16, 8))
fig.suptitle(
    f'The Multiple Linear Regression Skills Vs Salary\nR2 score: {r2_score(y_test, y_pred_test):.4f}',
    fontsize=14,
    fontweight='bold'
)

sns.heatmap(
    df.corr().round(2),
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    ax=axes[0, 0],
    linewidths=0.5,
    vmin=-1,
    vmax=1,
)
axes[0, 0].set_title("Correlation Heatmap\n(r=1.0 → Problem! )")

ax1 = axes[0, 1]
ax1.scatter(
    y_train,
    y_pred_train,
    s=80,
    alpha=0.7,
    color='orange',
    edgecolor='white',
    label='Training Data',
)
y_min, y_max = y.min(), y.max()
ax1.plot(
    [y_min, y_max],
    [y_min, y_max],
    alpha=0.6,
    lw=1.6,
    color='red',
    label='Regression Line',
)
ax1.set_xlabel('Actual Values', fontsize=10, fontweight='bold')
ax1.set_ylabel('Predicted Values', fontsize=10, fontweight='bold')
ax1.set_title('Actual Vs Predicted', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.4)

ax2 = axes[1, 0]
ax2.scatter(
    y_pred_test,
    residuals_test,
    s=80,
    alpha=0.7,
    color='green',
    edgecolors='white',
    label='Residuals',
)
ax2.axhline(y=0, lw=2, linestyle='--', color='red', alpha=0.7, label='Zero Errors')
ax2.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
ax2.set_ylabel('Residuals', fontsize=12, fontweight='bold')
ax2.set_title('Residuals Distribution', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.5)

ax3 = axes[1, 1]
ax3.hist(residuals_train, bins=25, edgecolor='white', color='violet', label='Residuals Distribution')
ax3.set_xlabel('Residuals', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax3.set_title('Residuals Distribution', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.savefig('Multiple Linear Regression.png', dpi=300)
print('Mlr save !')
plt.show()

