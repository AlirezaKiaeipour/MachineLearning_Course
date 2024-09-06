import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

class Perceptron:
    def __init__(self,epoch=1,learning_rate=0.01,bias=0.1):
        self.epochs = epoch
        self.learning_rate = learning_rate
        self.learning_rate_b = bias
        self.w = np.random.rand(1,1)
        self.b = np.random.rand(1,1)
        self.loss = []

    def fit(self,x_train,y_train):
        for epoch in range(1):
            for i in range(x_train.shape[0]):
                predict_train = np.matmul(x_train[i],self.w) + self.b
                error = y_train[i] - predict_train
                self.w += error * self.learning_rate * x_train[i]
                self.b += self.learning_rate * error
                pred = np.matmul(x_train,self.w)
                loss = np.mean(np.abs(y_train-pred))
                self.loss.append(loss)
    
    def predict(self,x_test):
        y_pred = np.matmul(x_test,self.w) + self.b
        return y_pred
    
    def mean_absolute_error(self,y_test , y_pred):
        error = y_test - y_pred
        return np.mean(np.abs(error))
    
    def mean_squared_error(self,y_test,y_pred):
        error = y_test - y_pred
        return np.mean(error**2)
    
    def r2score(self,y_test,y_pred):
        return metrics.r2_score(y_test,y_pred)
    
    def plot_error_rate(self):
        plt.plot(self.loss)
    
    def save_model(self):
        with open("weight.npy","wb") as f:
            pickle.dump(self.w,f)
        with open("bias.npy","wb") as f:
            pickle.dump(self.b,f)