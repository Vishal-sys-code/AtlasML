import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Loading Dataset
df = pd.read_csv('./datasets/placements.csv')

# Splitting X and y
X = df.iloc[:,0].values
y = df.iloc[:,1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
Implementation of Multiple Linear Regression

Beta = (X^T X)^-1 X^T y

Beta[0] => the intercept
Beta[1:] => the coefficients for each feature [i.e. slopes]
"""

class multiLinearRegression:
    def __init__(self, coefficient, intercept):
        self.coefficient = None
        self.intercept = None
    def forward(self, X_train, y_train):
        X_train = np.insert(X_train, 0, 1, axis = 1) # np.insert (array_name, array_index, value, axis)
        # Calculating the coefficients
        beta = np.linalg.inv(np.dot(X_train.T, X_train)).dot(X_train.T).dot(y_train)
        self.intercept = beta[0]
        self.coefficient = beta[1:]
        print("Coefficients are:", self.coefficient)
        print("Intercept is:", self.intercept)
    def predict(self, X_test):
        y_pred = np.dot(X_test, self.coefficient) + self.intercept
        return y_pred
    
mlr = multiLinearRegression()
mlr.fit(X_train, y_train)
y_pred = mlr.predict(X_test[0])
print("Predicted value from the Simple Linear Regression is: ", y_pred)