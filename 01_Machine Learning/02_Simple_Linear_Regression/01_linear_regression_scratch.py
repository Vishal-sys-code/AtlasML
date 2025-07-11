# Importing Libraries
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

# Implementation of Simple Linear Regression
"""
Equations are:
slope = Summation_from_1toN[(x_i - x_mean)(y_i - y_mean)] / Summation_from_1toN[(x_i - x_mean)^2]
intercept = y_mean - slope * x_mean

intercept is the bias, and slope is the weight.
"""

class SimpleLinearRegression:
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, X_train, y_train):
        numerator = 0
        denominator = 0
        for i in range(X_train.shape[0]):
            numerator = numerator + ((X_train[i] - X_train.mean()) * (y_train[i] - y_train.mean()))
            denominator = denominator + ((X_train[i] - X_train.mean()) * (X_train[i] - X_train.mean()))
        self.slope = numerator / denominator
        self.intercept = (y_train.mean() - self.slope * X_train.mean())
        print("Slope(m) is: ", self.slope)
        print("Intercept is: ", self.intercept)

    def predict(self, X_test):
        return self.slope * X_test + self.intercept # mX + b
    
lr = SimpleLinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test[0])
print("Predicted value from the Simple Linear Regression is: ", y_pred)