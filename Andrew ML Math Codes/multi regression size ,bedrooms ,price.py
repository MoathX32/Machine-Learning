import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'E:\\course training\\Regression\\multi\\T1\\multi reg data.txt'
data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

# rescaling data
data = (data - data.mean()) / data.std()

# # add ones column
data.insert(0, 'Ones', 1)
cols = data.shape[1]
x = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# convert to matrices and initialize theta
x = np.matrix(x.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0,0]))

def costf(x , y ,theta):
    z = np.power(((x * theta.T) - y), 2)
    return np.sum(z) / (2 * len(x))

# GD function
def gradientDescent(x, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    for i in range(iters):
        error = (x * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error, x[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(x)) * np.sum(term))
        theta = temp
        cost[i] = costf(x, y, theta)
    return theta, cost
# initialize variables for learning rate and iterations
alpha = 0.1
iters = 100
# perform gradient descent to "fit" the model parameters
theta0, cost = gradientDescent(x, y, theta, alpha, iters)
print('g = ' , theta0)

# get best fit line for Size vs. Price
d = np.linspace(data.Size.min(), data.Size.max(), 100)
f = theta0[0, 0] + (theta0[0, 1] * d)
print('f \n',f)
# draw the line for Size vs. Price
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(d, f, 'r', label='Prediction')
ax.scatter(data.Size, data.Price, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Size')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')
# get best fit line for Bedrooms vs. Price
d = np.linspace(data.Bedrooms.min(), data.Bedrooms.max(), 100)
f = theta0[0, 0] + (theta0[0, 1] * d)
# draw the line for Bedrooms vs. Price
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(d, f, 'r', label='Prediction')
ax.scatter(data.Bedrooms, data.Price, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Bedrooms')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')
# draw error graph
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')