import numpy as np
import matplotlib.pyplot as plt
    
    
class logistic_reg:
    def predict(self, x, theta):
        return (1 / (1 + np.exp(-x.dot(theta))))
    
    def predict1(self, x, theta):
        x1 = np.zeros(x.shape)
        x1[0, 0] = 1
        x1[0, 1:] = (x[0, 1:]-self.X_mean)/self.X_sigma
        return self.predict(x1, theta)
    
    def cost_function(self, X, Y, theta):
        m = X.shape[0]
        # print predict(X[1, :], theta)
        sum = 0
        for i in range(m):
            sum += Y[i][0] * np.log(self.predict(X[i, :], theta)) + (1 - Y[i][0]) * np.log(1 - self.predict(X[i, :], theta))
        return -sum / m
       
    # Read data from file
    def load_data(self, filename):
        f = open(filename, 'r')
        data = []
        for line in f:
            row = line.strip().split(',')
            data.append(row)
        
        m = len(data)
        col = len(data[0]) - 1
        X = np.zeros((m, col))
        Y = np.zeros((m, 1))
        for i in range(m):
            for j in range(col):
                X[i][j] = float(data[i][j])
            Y[i] = float(data[i][-1])
            
        return X, Y
    
    def map_feature(self, X):
        degree = 3
        vec = []
        for i in range(degree + 1):
            for j in range(degree + 1):
                if ((i + j <= degree)):
                    vec.append([i, j])
        x1 = X[:, 0]
        x2 = X[:, 1]
        X_ext = np.zeros((X.shape[0], len(vec)))
        for i in range(len(vec)):
            X_ext[:, i] = x1**vec[i][0] * x2**vec[i][1]
        return X_ext
    
    # feature normalization
    def normalize_feature(self, X):
        X = X.copy()
        self.X_mean = np.reshape(np.average(X[:, 1:], axis=0), (1, -1))
        X[:, 1:] -= self.X_mean
        self.X_sigma = np.reshape(np.average(X[:, 1:] ** 2, axis=0) ** 0.5, (1, -1))
        X = X.dot(np.diag(np.r_[1, self.X_sigma.ravel()] ** -1))
        return X
     
    def theta_unnormalized(self, theta):
        theta_unnorm = np.zeros(theta.shape)
        theta_unnorm[1:, 0] = theta[1:, 0].T / self.X_sigma
        theta_unnorm[0][0] = theta[0][0] - np.sum(theta[1:].T * self.X_mean * (self.X_sigma ** -1))
        #print 'sum = ', theta[1:].T * self.X_mean * (self.X_sigma ** -1)
        return theta_unnorm
        
    def gradient(self, X, Y, theta):
        m = X.shape[0]
        feature_num = len(X[0]) - 1
        h = range(m)
        for i in range(m):
            h[i] = self.predict(X[i, :], theta)
        
        grad = np.zeros((feature_num + 1, 1))
        for j in range(feature_num + 1):
            sum = 0
            for i in range(m):
               sum += X[i][j] * (Y[i][0] - h[i])
            grad[j][0] = sum[0]/m
        return grad
        
    def fit(self, X, Y):
        X = self.map_feature(X)
        X = self.normalize_feature(X)
        feature_num = len(X[0]) - 1
        iter_num = 1000
        theta = np.zeros((feature_num + 1, 1))
        #theta = np.reshape([1, -1, 10], (3, 1))
        alpha = 1
        J = np.zeros((1, iter_num))
        for k in range(iter_num):
            theta = theta + alpha * self.gradient(X, Y, theta)
            # print theta
            #J[k] = self.cost_function(X, Y, theta)
        theta = self.theta_unnormalized(theta)
        return theta, J
        
    def plot_result(self, theta):
        import matplotlib.pyplot as plt
        delta = 0.1
        x = np.arange(-3.0, 3.0, delta).reshape(-1, 1)
        y = np.arange(-2.0, 2.0, delta).reshape(-1, 1)
        Z = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                Z[i, j] = self.map_feature(np.r_[x[i], y[j]].reshape(1, -1)).dot(theta)
        
        plt.contour(x, y, Z, [0])
        plt.show()
        


        
