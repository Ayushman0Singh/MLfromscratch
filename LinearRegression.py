import numpy as np

def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2

class LinearRegression:
    # adding required init elements 
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        """
        Fits the linear regression model.
        
        Initializes the weights and bias, and runs gradient descent for the specified
        number of iterations to learn the weights and bias. 
        
        X: Training data inputs
        y: Training data outputs
        """
        # Initialize weights and bias
        self.weights = np.ones(X.shape[1])
        self.bias = 0
        # Gradient Descent
        for _ in range(self.num_iterations): # number for iterations for convergence
            predictions = np.dot(X,self.weights) + self.bias
            dw = 2/X.shape[0]*np.dot(X.T, predictions-y) 
            db = 2/X.shape[0]*np.sum(predictions-y)
            # Update weights and bias
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db


    def predict(self, X):
        # Make predictions using the learned weights and bias
        return np.dot(X,self.weights) + self.bias

# Testing
if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=10, random_state=4
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = LinearRegression(learning_rate=0.01, num_iterations=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    print("MSE:", mse)

    accu = r2_score(y_test, predictions)
    print("Accuracy:", accu)

    y_pred_line = regressor.predict(X)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.show()