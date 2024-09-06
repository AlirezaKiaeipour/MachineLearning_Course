import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

data = datasets.load_boston()

x_train = np.matrix(data.data[:,6:8])
y_train = data.target

learning_rate = 0.00001
w = np.random.rand(2,1)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
for i in range(x_train.shape[0]):
    predict_train = np.matmul(x_train[i],w)
    error = y_train[i] - predict_train
    w += x_train[i].T  * error * learning_rate 

    ax.clear()
    ax.scatter(x_train[:,0],x_train[:,1],y_train,c="orange")
    X = np.arange(x_train[:,0].min(),x_train[:,0].max())
    Y = np.arange(x_train[:,1].min(),x_train[:,1].max())
    X, Y = np.meshgrid(X, Y)
    Z = X * w[0]  + Y * w[1]
    surf = ax.plot_surface(X, Y, Z,alpha=0.5)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Age")
    ax.set_zlabel("Price")
    plt.pause(0.1)
plt.show()
