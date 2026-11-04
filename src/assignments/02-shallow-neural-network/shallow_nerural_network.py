import numpy as np
np.random.seed(42)

from sklearn.datasets import make_moons

def make_data(n, noise, seed):
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return X.T, y.reshape(1, -1)          # (2, n), (1, n)

X_train, Y_train = make_data(400,  0.2, 0)
X_test,  Y_test  = make_data(1000, 0.2, 99)