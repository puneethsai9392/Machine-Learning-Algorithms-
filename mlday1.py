import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LinearRegression
from sklearn.metrics         import mean_squared_error,r2_score,mean_absolute_error


np.random.seed(42)

X=np.random.randint(1,200,500).reshape(-1,1)
noise=np.random.randint(-700,600,500)
y=50+30*X.flatten()+noise

df=pd.DataFrame({'speed_km':X.flatten(),'cost_km':y})


print('First 5 rows of the dataset:')
print(df.head())

print('\nDataset information:')
print(df.info())

print('\nDataset description:')
print(df.describe())

print('\nMissing values in the dataset:')
print(df.isnull().sum())

print('\nCorrelation matrix:')
print(df.corr())

#model tarining

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)

y_test_pred=model.predict(X_test)
y_train_pred=model.predict(X_train)

def evaluate(y_true,y_pred):
    mse=mean_squared_error(y_true,y_pred)
    r2=r2_score(y_true,y_pred)
    mae=mean_absolute_error(y_true,y_pred)

    print('\nModel Performance Metrics:')
    print(f'Mean Squared Error: {mse}')
    print(f'R² Score: {r2}')
    print(f'Mean Absolute Error: {mae}')
    print(f'root Mean Squared Error: {np.sqrt(mse)}')
evaluate(y_test,y_test_pred)
evaluate(y_train,y_train_pred)


print(f'model Coefficients: {model.coef_[0]:.2f}')
print(f'model Intercept: {model.intercept_:.2f}')

speed_km=int(input('Enter the speed in km/h to predict the cost: '))
result=model.predict([[speed_km]])
print(f'The predicted cost for speed {speed_km} km/h is: {result[0]:.2f}')

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle(
    f"Simple Linear Regression Model: Speed vs Cost\nR² Score: {r2_score(y_test, y_test_pred):.2f}",
    fontsize=16,
    fontweight='bold'
)


axes1=axes[0]
axes1.scatter(X_test, y_test, color='orange', alpha=0.5,s=80,edgecolors='black',label='Training data')
X_line=np.linspace(X.min(),X.max(),100).reshape(-1,1)
axes1.plot(X_line, model.predict(X_line), lw=2,color='red', label='Regression line') 
axes1.set_xlabel('Speed (km/h)')    
axes1.set_ylabel('Cost (currency)')
axes1.set_title('Training Data and Regression Line')    
axes1.legend()
axes1.grid(True)
34

axes2=axes[1]
axes2.scatter(y_test, y_test_pred, color='green', alpha=0.5,s=80,edgecolors='black',label='Testing data')
axes2.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, color='red', label='Perfect Prediction')
axes2.set_xlabel('Actual Cost')
axes2.set_ylabel('Predicted Cost')
axes2.set_title('Actual vs Predicted Cost')
axes2.legend()
axes2.grid(True,alpha=0.5)


residulas=y_test-y_test_pred
axes3=axes[2]
axes3.scatter(y_test_pred,residulas, color='blue', alpha=0.5,s=80,edgecolors='black',label='Residuals')
axes3.axhline(y=0, color='red', linestyle='--', lw=2, label='Zero Error Line')
axes3.set_xlabel('Predicted Cost')      
axes3.set_ylabel('Residuals')
axes3.set_title('Residuals vs Predicted Cost')
axes3.legend()
axes3.grid(True,alpha=0.5)

axes4=axes[3]
axes4.hist(residulas, bins=20, color='blue', alpha=0.5, edgecolor='black')
axes4.set_xlabel('Residuals')   
axes4.set_ylabel('Frequency')
axes4.set_title('Distribution of Residuals')    
axes4.grid(True,alpha=0.5)


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('linear_regression_analysis.png', dpi=300)
print('Analysis plot saved as linear_regression_analysis.png')

