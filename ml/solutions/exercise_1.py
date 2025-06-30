import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

np.random.seed(1234)
# create two gaussians
A = np.random.multivariate_normal(mean=[1, 1], cov=[[2, 1], [1, 2]], size=200)
B = np.random.multivariate_normal(mean=[-2, -2], cov=[[2, 0], [0, 2]], size=200)

# get them into proper matrix form
X = np.vstack([A, B])
Y = np.hstack([np.zeros(len(A)), np.ones(len(B))])

# train the linear regressor and save the coefficents
reg = linear_model.LinearRegression()
reg.fit(X, Y)
b_1, b_2 = reg.coef_
b_0 = reg.intercept_

# solve the function y = b_0 + b_1*X_1 + b_2 * X_2 for X2
x1s = np.linspace(-8, 8)
x2s = (0.5 - b_0 - b_1 * x1s) / b_2


plt.scatter(A[:, 0], A[:, 1], s=25, color="dodgerblue", label="True class A")
plt.scatter(B[:, 0], B[:, 1], s=25, color="limegreen", label="True class B")

plt.plot(x1s, x2s, color="gray", linestyle="--")

plt.fill_between(x1s, x2s, 10, color="dodgerblue", alpha=0.07)
plt.fill_between(x1s, x2s, -10, color="limegreen", alpha=0.07)
plt.grid()
plt.xlabel("X1")
plt.ylabel("X2")
plt.margins(x=0, y=0)
plt.xlim([-8, 8])
plt.ylim([-8, 8])
plt.legend()
None
