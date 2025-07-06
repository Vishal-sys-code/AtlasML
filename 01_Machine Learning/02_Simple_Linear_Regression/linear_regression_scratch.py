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

"""

class SimpleLinearRegression:
    def __init__():
        pass
    def fit():
        pass
    def predict():
        pass