import numpy as np

class BaseRegression:

    def __init__(self, learning_rate=0.001, num_iterations=1000) -> None:
        self.learning_rate = learning_rate 
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        self.weights = np.ones(X.shape[1])
        self.bias = 0
        n_samples,n_features = X.shape
        

        for i in range(self.num_iterations): # applying gradient descent for num_iterations

            pred = self._approximation(X,self.weights,self.bias)
            
            dw = (1/n_samples) * np.dot(pred-y, X)
            self.weights = self.weights - dw*self.learning_rate # weight adjustment

            db =  (1/n_samples) * np.sum(pred - y) 
            self.bias = self.bias - db*self.learning_rate

    def predict(self, X):
        return self._predict(X, self.weights, self.bias)

    def _predict(self, X, w, b):
        raise NotImplementedError

    def _approximation(self, X, w, b):
        raise NotImplementedError


class LinearRegression(BaseRegression):
    def _approximation(self, X, w, b):
        return np.dot(X,w) + b

    def _predict(self, X,w,b):
        return np.dot(X, w) + b
    

class LogisticRegression(BaseRegression):
    def _approximation(self, X, w, b):
        linear_model = np.dot(X, w) + b
        return self._sigmoid(linear_model)

    def _predict(self, X, w, b):
        linear_model = np.dot(X, w) + b
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self,x):
        return 1/(np.exp(-x) + 1)
        


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score

    X,y = datasets.make_regression(n_samples=100, n_features=1, noise=10, random_state=4)
    X_train,X_test, y_train, y_test = train_test_split(X,y)
    model = LinearRegression(learning_rate=0.01, num_iterations=1000)
    model.fit(X_train,y_train)
    y_predicted = model.predict(X_test)

    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    mse = mean_squared_error(y_test, y_predicted)
    print("MSE:", mse)

    accu = r2_score(y_test, y_predicted)
    print("Accuracy:", accu)

    y_pred_line = model.predict(X)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.show()

    # LOGISTIC REGRESSION
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = LogisticRegression(learning_rate=0.0001, num_iterations=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)
    print("Logistic reg classification accuracy:", accuracy(y_test, predictions))