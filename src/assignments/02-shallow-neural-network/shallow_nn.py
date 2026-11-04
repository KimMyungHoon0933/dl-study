import numpy as np
np.random.seed(42)

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import sys

sys.path.insert(
    0,
    r"../01-logistic-regression"
)
from logistic_regression import initialize, optimize, predict as logistic_predict   

# 데이터 불러오기
def make_data(n, noise, seed):
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return X.T, y.reshape(1, -1)          # (2, n), (1, n)




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


# 파라미터 초기화
def initialize_parameters(n_x, n_h, n_y):
    W1=np.random.randn(n_h, n_x)*0.01
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y, n_h)*0.01
    b2=np.zeros((n_y,1))

    parameters={
        "W1":W1,
        "W2":W2,
        "b1":b1,
        "b2":b2
    }

    return parameters 


#활성화 함수

def sigmoid(z):
    #예상 입력: z=(1,m)
    z=np.array(z)
    A=np.zeros_like(z, dtype=float)
    positive = (z >= 0)
    
    A[positive]=1 / (1 + np.exp(-z[positive]))

    exp_z = np.exp(z[~positive])
    A[~positive] = exp_z / (1 + exp_z)

    return A #출력 A.shape=(1,m)
def tanh(z):
    z = np.asarray(z, dtype=float)
    exp_z = np.exp(-2 * np.abs(z))
    return np.sign(z) * (1 - exp_z) / (1 + exp_z)


#순전파
def forward_propagation(X, parameters):
    W1=parameters["W1"]
    W2=parameters["W2"]
    b1=parameters["b1"]
    b2=parameters["b2"]
    Z1=W1@X +b1
    A1=tanh(Z1)
    Z2=W2@A1 +b2
    A2=sigmoid(Z2)
    cache={
        "Z1":Z1,
        "Z2":Z2,
        "A1":A1,
        "A2":A2
    }
    return A2,cache


#비용함수
def compute_cost(Z2, Y):
    L = np.logaddexp(0, Z2) - Y * Z2
    J=np.sum(L)/Z2.shape[1]
    return J

#역전파
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]

    dZ2=cache["A2"]-Y
    dW2=(dZ2@cache["A1"].T)/m
    db2=np.sum(dZ2,axis=1,keepdims=True)/m 
    dZ1=(parameters["W2"].T@dZ2) * (1-cache["A1"]**2)

    dW1=dZ1@X.T/m
    db1=np.sum(dZ1,axis=1,keepdims=True)/m 

    grads={
        "dZ2":dZ2,
        "dW2":dW2,
        "db2":db2,
        "dZ1":dZ1,
        "dW1":dW1,
        "db1":db1,
    }
    return grads

#가중치 업데이트
def update_parameters(parameters, grads, learning_rate):
    W1=np.asarray(parameters["W1"]).copy()
    W2=np.asarray(parameters["W2"]).copy()
    b1=np.asarray(parameters["b1"]).copy()
    b2=np.asarray(parameters["b2"]).copy()

    W2 -= learning_rate* grads["dW2"]
    b2 -= learning_rate* grads["db2"]

    W1 -= learning_rate* grads["dW1"]
    b1 -= learning_rate* grads["db1"]

    parameters={
        "W1":W1,
        "W2":W2,
        "b1":b1,
        "b2":b2
    }
    return parameters


#예측
def predict(parameters, X):
    A2,_=forward_propagation(X,parameters)
    predict= (A2 >= 0.5).astype(int)
    return predict


#최종 모델 함수
def nn_model(X, Y, n_h, num_iterations, learning_rate, X_test, Y_test):
    
    parameters=initialize_parameters(X.shape[0],n_h,Y.shape[0])
    history = {
        "cost_list": [],
        "test_cost_list": [],
        "grads_list": [],
        "accuracy_iterations": [],
        "train_accuracy": [],
        "test_accuracy": [],
    }

    for i in range(num_iterations):
        #학습
        A2,cache=forward_propagation(X ,parameters)
        cost_function=compute_cost(cache["Z2"],Y)
        grads=backward_propagation(parameters,cache,X,Y)
        parameters=update_parameters(parameters,grads,learning_rate)
        
        #기록
        A2_test, cache_test = forward_propagation(X_test, parameters)

        history["cost_list"].append(compute_cost(cache["Z2"], Y))
        history["test_cost_list"].append(
            compute_cost(cache_test["Z2"], Y_test)
        )
        history["grads_list"].append(grads)
        
        #정확도 계산
        predict_train = predict(parameters, X)
        predict_test = predict(parameters, X_test)

        history["accuracy_iterations"].append(i + 1)
        history["train_accuracy"].append(np.mean(predict_train == Y))
        history["test_accuracy"].append(np.mean(predict_test == Y_test))


    return parameters,history






def main():
    # 데이터 크기 
    X_train, Y_train = make_data(400,  0.2, 0) #x[0] x좌표 x[1]이 y좌표
    #(2,400) (1,400)
    X_test,  Y_test  = make_data(1000, 0.2, 99)
    #(2,1000) (1,1000)
    print(X_train.shape,X_test.shape)

    #데이터 시각화 x_train
    plt.scatter(X_train[0], X_train[1], c=Y_train.ravel(), cmap="coolwarm", s=15)
    plt.title("train")
    plt.tight_layout()
#    plt.show()

    #데이터 시각화 x_test
    plt.scatter(X_test[0], X_test[1], c=Y_test.ravel(), cmap="coolwarm", s=15)
#    plt.title("test")
    plt.tight_layout()
#    plt.show()

    # 로지스틱 회귀 test
    def predict_fn(X_grid):
        w, b = initialize(X_train.shape[0])

        params, _, _ = optimize(
            w, b, X_train, Y_train,
            num_iterations=2000,
            learning_rate=0.05
        )

        return logistic_predict(params["w"], params["b"], X_grid)

    plot_decision_boundary(predict_fn, X_train, Y_train)
    #plt.show()

    #initialize_parameters test
    parameters=initialize_parameters(X_train.shape[0],3,1)
    for name, value in parameters.items():
        print(f"{name}: {value.shape}")

    #forward test
    A2,cache=forward_propagation(X_train,parameters)
    print(A2.shape)

    #cost_function test
    cost_function=compute_cost(cache["Z2"],Y_train)
    print(cost_function)

    #backpropagation test
    grads=backward_propagation(parameters,cache,X_train,Y_train)
    for name, value in grads.items():
        print(f"{name}: {value.shape}")

    parameters=update_parameters(parameters,grads,0.001)
    for name, value in parameters.items():
        print(f"{name}: {value}")

    #predict test
    acc=predict(parameters,X_train)
    print(acc.shape)

    #nn_model test
    parameters,history=nn_model(X_train,Y_train,3,100,0.01,X_test,Y_test)

if __name__ == "__main__":
    main()