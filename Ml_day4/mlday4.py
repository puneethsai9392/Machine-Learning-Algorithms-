import warnings 
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model    import LinearRegression 
from sklearn.preprocessing   import StandardScaler,PolynomialFeatures
from sklearn.metrics         import mean_squared_error,mean_absolute_error,r2_score,root_mean_squared_error
from sklearn.pipeline        import Pipeline


df=pd.read_csv(r'C:\Users\Hp\Documents\simple_linear_day1\multiple_linear_regression_dataset.csv')


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
print(df.corr()['income'].sort_values(ascending=False))

print(f"\nThe shape of the dataset is: {df.shape}")



X=df.drop(columns=['income'])
y=df['income']
print(X.head())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

lrm=Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

lrm.fit(X_train,y_train)
y_train_pred=lrm.predict(X_train)
y_test_pred=lrm.predict(X_test)

def evalute(y_true,y_pred,label):
    mae=mean_absolute_error(y_true,y_pred)
    mse=mean_squared_error(y_true,y_pred)
    rmse=np.sqrt(mse)
    r2=r2_score(y_true,y_pred)
    print('-'*60)
    print('Model evalution metrics -->',label)
    print('-'*60)
    print('mean_absolute_error     :',mae)
    print('mean_squared-error      :',mse)
    print('root_mean_squared _eror :',rmse)
    print('r2_score                :',r2)
    return mae,mse,rmse,r2

test_mae, test_mse, test_rmse, test_r2 = evalute(y_test, y_test_pred, 'Test_data')
train_mae, train_mse, train_rmse, train_r2 = evalute(y_train, y_train_pred, 'Train_data')
print()
print('-'*60)
for fet, coef_ in zip(X.columns, lrm.named_steps['model'].coef_):
    print(f'{fet:<23}: {coef_}')
print(f"the model intercept is      :{lrm.named_steps['model'].intercept_:.2f}")
print('-'*60)
test_residuals  = y_test - y_test_pred
train_residuals = y_train - y_train_pred

print('Test residuals  mean :', test_residuals.mean())
print('Test residuals  std  :', test_residuals.std())
print('Train residuals mean :', train_residuals.mean())
print('Train residuals std  :', train_residuals.std())


from scipy import stats

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    f'Multiple Linear Regression — Income Prediction\nR2 Score (Test): {r2_score(y_test, y_test_pred):.4f}',
    fontsize=18,
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
axes[0, 0].set_title("1. Correlation Heatmap", fontsize=12, fontweight='bold')

ax2 = axes[0, 1]
ax2.scatter(y_train, y_train_pred, s=70, alpha=0.7, color='orange', edgecolor='white', label='Training Data')
y_min, y_max = y.min(), y.max()
ax2.plot([y_min, y_max], [y_min, y_max], alpha=0.6, lw=1.6, color='red', label='Ideal Fit')
ax2.set_xlabel('Actual Values', fontsize=10, fontweight='bold')
ax2.set_ylabel('Predicted Values', fontsize=10, fontweight='bold')
ax2.set_title('2. Actual Vs Predicted (Train)', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.4)

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

ax6 = axes[1, 2]
stats.probplot(test_residuals, dist="norm", plot=ax6)
ax6.get_lines()[0].set_markerfacecolor('teal')
ax6.get_lines()[0].set_markeredgecolor('white')
ax6.get_lines()[0].set_markersize(6)
ax6.get_lines()[1].set_color('red')
ax6.set_title('6. Q-Q Plot Of Test Residuals', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.5)

plt.tight_layout(rect=[0, 0.02, 1, 0.94])
plt.savefig('Multiple Linear Regression Data SET.png', dpi=300, bbox_inches='tight')
print('MLR dashboard saved!')
plt.show()