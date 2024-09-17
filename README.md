<div id="header" align="center">
  <img src="https://media.giphy.com/media/M9gbBd9nbDrOTu1Mqx/giphy.gif" width="100"/>
</div>

## ML Implementations from scratch
Small Implementation of a linear regression, logistic regression and Decision tree model from scratch using Python.

**Introduction**
This code offers the following functionalities:

- Custom `r2_score` function to calculate the R-squared value (coefficient of determination).
- `LinearRegression` class for linear regression:
    - Configurable learning rate and number of iterations.
    - `fit` method to train the model on a dataset.
    - `predict` method to make predictions.
- Testing script demonstrating usage:
    - Creates synthetic data.
    - Trains and fits the model.
    - Makes predictions and evaluates performance.
    - Visualizes results.

**Usage**

1. **Import and initialize:**

   ```python
   from regression import LinearRegression

   model = LinearRegression(learning_rate=0.001, num_iterations=1000)

2. **Use the Fit and predict Methods**
    ```python
    X_train, y_train = ...  # Your training data
    model.fit(X_train, y_train)
    predictions = model.predict(X_new)

# Dependencies
- numpy
- sklearn (used for testing)
- matplotlib (used for testing)

