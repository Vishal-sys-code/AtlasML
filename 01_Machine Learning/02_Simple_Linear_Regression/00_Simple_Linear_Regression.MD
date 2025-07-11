# Simple Linear Regression
- Regression means that it is a relationship between the input and the output variables.
- Independent Variable = X-axis and Dependent Variable = Y-axis
- Simple Linear Regression is a simple algorithm that gives the relationship between the independent and the dependent variables.
- Y = mX + b [independent_variable = slope * dependent_variable + intercept]
- By default sklearn has implemented the LinearRegression in the closed form solution i.e. Ordinary Least Squares (OLS).
- There is an another class called SGDRegression when there is a non-closed form of solution then we use this.

**Closed Form Solution:** Finite set of function using airthmetic operations. -> 1D <br>
**Non-Closed Form Solution:** Finite set of function using differentiation and integration. -> For higher dimensions<br>

## Simple Linear Regression using Maths or Normal Equation

$$
\text{slope} = \frac{\sum_{i=1}^{N} (y_i - \bar{y})(x_i - \bar{x})}{\sum_{i=1}^{N} (x_i - \bar{x})^2}
$$

$$
\text{intercept} = \bar{y} - \text{slope} \cdot \bar{x}
$$

## Simple Linear Regression using Library

```
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(y_pred)
```
## Loss function of Simple Linear Regression

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \left( y_i - (\hat{y}_i) \right)^2
$$

Where:

- \( y_i \) is the true value
- \( \hat{y}_i = mx_i + b \) is the predicted value using the learned model
- \( m \) is the slope
- \( b \) is the intercept
- \( N \) is the number of data points