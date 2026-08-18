import numpy as np

class GenerateData:
    def linear_data(self, num=100):
        X = np.random.randn(num, 2)
        Y = (X[:, 0] + X[:, 1] > 0).astype(int)                   # ((X[:,0] > 0) ^ (X:, 1) > 0)).astype(int)
        # print(Y)

        return X, Y

    def non_linear_data(self, num=100):
            X = np.random.randn(num, 2)
            Y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
            # print(Y)
            
            return X, Y
