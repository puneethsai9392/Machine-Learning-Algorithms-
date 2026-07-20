import warnings 
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

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
