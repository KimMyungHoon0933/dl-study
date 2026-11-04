import numpy as np
np.random.seed(42)

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import sys

sys.path.insert(
    0,
    r"../01-logistic-regression"
)
from logistic_regression import initialize, optimize, predict   

# 데이터 불러오기
def make_data(n, noise, seed):
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return X.T, y.reshape(1, -1)          # (2, n), (1, n)

def tanh(z):
    z = np.asarray(z, dtype=float)
    exp_z = np.exp(-2 * np.abs(z))
    return np.sign(z) * (1 - exp_z) / (1 + exp_z)


#로지스틱 회귀 결과 시각화
def plot_decision_boundary(predict_fn, X, Y, title=""):
    x_min, x_max = X[0].min()-0.5, X[0].max()+0.5
    y_min, y_max = X[1].min()-0.5, X[1].max()+0.5
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = predict_fn(np.c_[xx.ravel(), yy.ravel()].T).reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.4)
    plt.scatter(X[0], X[1], c=Y.ravel(), s=12, cmap=plt.cm.Spectral)
    plt.title(title)

def predict_fn(X_grid):
    w, b = initialize(X_train.shape[0])

    params, _, _ = optimize(
        w, b, X_train, Y_train,
        num_iterations=2000,
        learning_rate=0.05
    )

    return predict(params["w"], params["b"], X_grid)

def initialize_parameters(n_x, n_h, n_y):
    W1=np.random.randn(n_h, n_x)
    b1=np.zeros(n_h)
    W2=np.random.randn(n_y, n_h)
    b2=np.zeros(n_y)

    parameters={
        "W1":W1,
        "W2":W2,
        "b1":b1,
        "b2":b2
    }

    return parameters 

def forward_propagation(X, parameters):
    W1=parameters["W1"]
    W2=parameters["W2"]
    b1=parameters["b1"]
    b2=parameters["b2"]
    Z1=W1@X +b1
    A1=tanh(Z1)
    Z2=W2@A1 +b2
    A2=tanh(Z2)
    cache={
        "Z1":Z1,
        "Z2":Z2,
        "A1":A1,
        "A2":A2
    }
    return A2,cache

def main():
    # 데이터 크기 
    X_train, Y_train = make_data(400,  0.2, 0) #x[0] x좌표 x[1]이 y좌표
    #(2,400) (1,400)
    X_test,  Y_test  = make_data(1000, 0.2, 99)
    #(2,1000) (1,1000)
    print(X_train.shape,X_test.shape)

    #데이터 시각화 x_train
#    fig ,axes = plt.subplots(1,5,figsize=(10,2))
#    for i, ax in enumerate(axes):
#        ax.scatter(X_train[0], X_train[1], c=Y_train.ravel(), cmap="coolwarm", s=15)
#        ax.set_title("train")
#        ax.set_aspect("equal")

#    plt.tight_layout()
#    plt.show()

    #데이터 시각화 x_test
#    fig ,axes = plt.subplots(1,5,figsize=(10,2))
#    for i, ax in enumerate(axes):
#        ax.scatter(X_test[0], X_test[1], c=Y_test.ravel(), cmap="coolwarm", s=15)
#        ax.set_title("test")
#        ax.set_aspect("equal")

#   plt.tight_layout()
#   plt.show()

    # 로지스틱 회귀 test
    def predict_fn(X_grid):
        w, b = initialize(X_train.shape[0])

        params, _, _ = optimize(
            w, b, X_train, Y_train,
            num_iterations=2000,
            learning_rate=0.05
        )

        return predict(params["w"], params["b"], X_grid)

    plot_decision_boundary(predict_fn, X_train, Y_train)
    plt.show()




if __name__ == "__main__":
    main()