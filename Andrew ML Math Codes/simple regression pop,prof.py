import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = ('E:\\course training\\Regression\\T1\\T1 data.txt')
data = pd.read_csv(path , header=None ,names=('population','profits'))
data.plot(kind='scatter',x='population',y='profits',figsize=(5,5))

data.insert(0 ,'ones', 1)
cols = data.shape[1]
x = np.matrix(data.iloc[:,0:cols-1])
y = np.matrix(data.iloc[:,cols-1:cols])
theta = np.matrix(np.array([0,0]))

# cost function
def costf(x , y ,theta):
    z = np.power(((x * theta.T) - y), 2)
    return np.sum(z) / (2 * len(x))
# print('computeCost(X, y, theta) = ' , cost_f(X, y, theta))

# GD function
def gd(x ,y ,theta ,alpha ,iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost =np.zeros(iters)
    for i in range (iters):
        error = ((x*theta.T)-y)
        for j in range (parameters):
            term = np.multiply(error, x[: ,j])
            temp[0 ,j] = theta[0 ,j] - ((alpha /len(x))*np.sum(term))
        theta = temp
        cost[i] = costf(x,y,theta)
    return theta , cost
alpha = 0.01
iters = 1000
theta1 ,cost = gd(x ,y ,theta ,alpha ,iters)

d = np.linspace(data.population.min() ,data.population.max() ,100)
f = theta1[0, 0] + (theta1[0, 1] * d)

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(d, f, 'r', label='Prediction')
ax.scatter(data.population, data.profits, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')

# draw error graph
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')