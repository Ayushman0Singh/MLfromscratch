from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression
import numpy as np

def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2
    

n_features = 1
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# visulize the data
fig = plt.figure(figsize=(8,6))
plt.scatter(X,y, s=10)

lr = LinearRegression()

lr.fit(X_train, y_train)

plt.plot(X, lr.predict(X), color='red', linewidth=0.3, label='Regression line')
# Show the plot

plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()