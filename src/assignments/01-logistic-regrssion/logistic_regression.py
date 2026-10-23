import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
mask = (digits.target == 0) | (digits.target == 1)
X, y = digits.data[mask], digits.target[mask]        # (360, 64), (360,)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)

X_train = X_tr.T / 16.0        # (64, 288)
X_test  = X_te.T / 16.0        # (64, 72)
Y_train = y_tr.reshape(1, -1)  # (1, 288)
Y_test  = y_te.reshape(1, -1)  # (1, 72)

print(X_train.shape)