<div id="header" align="center">
  <img src="https://media.giphy.com/media/M9gbBd9nbDrOTu1Mqx/giphy.gif" width="100"/>
</div>

## LinearRegression.py: Simple Linear Regression Implementation

This code implements a basic linear regression model from scratch using Python.

**Introduction**

This code offers the following functionalities:

- Custom `r2_score` function to calculate the R-squared value (coefficient of determination).
- `LinearRegression` class for linear regression:
    - Configurable learning rate and number of iterations.
    - `fit` method to train the model on a dataset.
    - `predict` method to make predictions.
- Testing script demonstrating usage:
    - Creates synthetic data.
    - Splits data into training and testing sets.
    - Trains and fits the model.
    - Makes predictions and evaluates performance.
    - Visualizes results.

**Usage**

1. **Import and initialize:**

   ```python
   import LinearRegression

   model = LinearRegression.LinearRegression(learning_rate=0.01, num_iterations=1000)

2. **Fit and prediction Methods**
    ```python
    X_train, y_train = ...  # Your training data
    model.fit(X_train, y_train)
    predictions = model.predict(X_new)

# Dependencies
-numpy
-sklearn (used for testing)
-matplotlib (used for testing)

