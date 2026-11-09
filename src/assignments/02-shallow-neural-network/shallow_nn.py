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
from logistic_regression import propagate as logistic_propagate

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
    W1=np.random.randn(n_h, n_x)*0.5
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y, n_h)*0.5
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
def nn_model(X, Y, n_h, num_iterations, learning_rate, X_test, Y_test,initializer=None,hidden_activate=None):

    if initializer is None:
        initializer = initialize_parameters
    
    if hidden_activate is None:
        hidden_activate = tanh

    parameters=initializer(X.shape[0],n_h,Y.shape[0])
    history = {
        "cost_list": [],
        "test_cost_list": [],
        "grads_list": [],
        "train_accuracy": [],
        "test_accuracy": [],
        "accuracy_iterations":[]
    }

    for i in range(num_iterations):
        #학습
        A2,cache=forward_propagation(X ,parameters)
        cost_function=compute_cost(cache["Z2"],Y)

        grads=backward_propagation(parameters,cache,X,Y)
        parameters=update_parameters(parameters,grads,learning_rate)
        
        #기록
        A2,cache=forward_propagation(X ,parameters)
        cost_function=compute_cost(cache["Z2"],Y)
        A2_test, cache_test = forward_propagation(X_test, parameters)
        test_cost_function=compute_cost(cache_test["Z2"], Y_test)
        history["cost_list"].append(compute_cost(cache["Z2"], Y))
        history["test_cost_list"].append(
            test_cost_function
        )
        history["grads_list"].append(grads)
        
        #정확도 계산
        if((i+1)%100==0):
            predict_train = predict(parameters, X)
            predict_test = predict(parameters, X_test)

            history["accuracy_iterations"].append(i + 1)
            history["train_accuracy"].append(np.mean(predict_train == Y))
            history["test_accuracy"].append(np.mean(predict_test == Y_test))


    return parameters,history


#비용 함수

def cost_function_plot(history,parameters):
    n_h=parameters["W1"].shape[0]
    horizontal = np.arange(1, len(history["cost_list"]) + 1)
    plt.plot(horizontal, history["cost_list"], label="train")
    plt.plot(horizontal, history["test_cost_list"], label="test")

    plt.title(f"n_h={n_h}")
    plt.xlabel("num_iterations")
    plt.ylabel("cost")
    plt.yscale("log")
    plt.legend()



def accuracy_plot(history,parameters):
    n_h=parameters["W1"].shape[0]

    plt.plot(history["accuracy_iterations"], history["train_accuracy"], label="train")
    plt.plot(history["accuracy_iterations"], history["test_accuracy"], label="test")

    plt.title(f"n_h={n_h}")
    plt.xlabel("num_iterations")
    plt.ylabel("accuracy")
    plt.legend()





# 기울기 검증
def flatten_parame(parameters):
    W1_reshape=np.asarray(parameters["W1"]).flatten()
    W2_reshape=np.asarray(parameters["W2"]).flatten()
    b1_reshape=np.asarray(parameters["b1"]).flatten()
    b2_reshape=np.asarray(parameters["b2"]).flatten()
    theta=np.concat([W1_reshape,b1_reshape,W2_reshape,b2_reshape])

    return theta

def flatten_grads(grads):
    W1_reshape=np.asarray(grads["dW1"]).flatten()
    W2_reshape=np.asarray(grads["dW2"]).flatten()
    b1_reshape=np.asarray(grads["db1"]).flatten()
    b2_reshape=np.asarray(grads["db2"]).flatten()
    theta=np.concat([W1_reshape,b1_reshape,W2_reshape,b2_reshape])

    return theta

def gradient_checks_test(X,Y,parameters):
    from utils.grad_check import grad_check
    X_check = X[:, :10]
    Y_check = Y[:, :10]
    
    _, cache = forward_propagation(X_check,parameters)
    cost =compute_cost(cache["Z2"],Y_check)
    grads=backward_propagation(parameters, cache, X_check, Y_check)

    analytic_grad = flatten_grads(grads)

    def theta_cost_function(theta):
        restored_parameters = {}
        start = 0

        for name in ("W1", "b1", "W2", "b2"):
            size = parameters[name].size
            shape = parameters[name].shape

            restored_parameters[name] = theta[start:start + size].reshape(shape)
            start += size

        _, cache = forward_propagation(X_check, restored_parameters)
        return compute_cost(cache["Z2"], Y_check)
            

    theta=flatten_parame(parameters)
    error=grad_check(theta_cost_function,theta,analytic_grad,epsilon=1e-5)

    return error

#experiment c 
def forward_propagation_C(X, parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    Z1 = W1 @ X + b1
    A1 = Z1
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }
    return A2, cache


def backward_propagation_C(parameters, cache, X, Y):
    m = X.shape[1]

    dZ2 = cache["A2"] - Y
    dW2 = (dZ2 @ cache["A1"].T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = parameters["W2"].T @ dZ2
    dW1 = (dZ1 @ X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {
        "dZ2": dZ2,
        "dW2": dW2,
        "db2": db2,
        "dZ1": dZ1,
        "dW1": dW1,
        "db1": db1
    }
    return grads


def predict_C(parameters, X):
    A2, _ = forward_propagation_C(X, parameters)
    return (A2 >= 0.5).astype(int)


def nn_model_C(X, Y, n_h, num_iterations, learning_rate,
               X_test, Y_test, initializer=None):
    if initializer is None:
        initializer = initialize_parameters

    parameters = initializer(X.shape[0], n_h, Y.shape[0])

    history = {
        "cost_list": [],
        "test_cost_list": [],
        "grads_list": [],
        "train_accuracy": [],
        "test_accuracy": [],
        "accuracy_iterations": []
    }

    for i in range(num_iterations):
        #학습
        _, cache = forward_propagation_C(X, parameters)
        grads = backward_propagation_C(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate)

        #기록
        _, cache = forward_propagation_C(X, parameters)
        _, cache_test = forward_propagation_C(X_test, parameters)

        history["cost_list"].append(compute_cost(cache["Z2"], Y))
        history["test_cost_list"].append(
            compute_cost(cache_test["Z2"], Y_test)
        )
        history["grads_list"].append(grads)

        #정확도 계산
        if (i + 1) % 100 == 0:
            predict_train = predict_C(parameters, X)
            predict_test = predict_C(parameters, X_test)

            history["accuracy_iterations"].append(i + 1)
            history["train_accuracy"].append(np.mean(predict_train == Y))
            history["test_accuracy"].append(np.mean(predict_test == Y_test))

    return parameters, history

#experiment D
def logistic_hessian(X, w, b):
    m = X.shape[1]

    a = sigmoid(w.T @ X + b)
    X_tilde = np.vstack([X, np.ones((1, m))])
    D = np.diag((a * (1 - a)).ravel())

    H = (X_tilde @ D @ X_tilde.T) / m

    eigenvalues = np.linalg.eigvalsh(H)
    L = float(eigenvalues[-1])
    limit = 2 / L if L > 0 else np.inf

    return H, L, limit

def main():
    # 데이터 크기 
    X_train, Y_train = make_data(400,  0.2, 0) #x[0] x좌표 x[1]이 y좌표
    #(2,400) (1,400)
    X_test,  Y_test  = make_data(1000, 0.2, 99)
    #(2,1000) (1,1000)
    print(X_train.shape,X_test.shape)

    #데이터 시각화 x_train
#    plt.scatter(X_train[0], X_train[1], c=Y_train.ravel(), cmap="coolwarm", s=15)
#    plt.title("train")
#    plt.tight_layout()
#    plt.show()

    #데이터 시각화 x_test
#    plt.scatter(X_test[0], X_test[1], c=Y_test.ravel(), cmap="coolwarm", s=15)
#    plt.title("test")
#    plt.tight_layout()
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

    predict_train=predict_fn(X_train)
    train_accuracy=np.mean(predict_train == Y_train)
    print("훈련 정확도: ",train_accuracy)
    
    predict_test=predict_fn(X_test)
    test_accuracy=np.mean(predict_test == Y_test)
    print("시험 정확도: ",test_accuracy)

##    plot_decision_boundary(predict_fn, X_train, Y_train)
#    plt.show()

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
    parameters,history=nn_model(X_train,Y_train,4,20000,0.2,X_test,Y_test)


    cost_function_plot(history,parameters)
    plt.show()

    accuracy_plot(history,parameters)
    plt.show()


    def predict_fn(X_grid):
        return predict(parameters, X_grid)

    plot_decision_boundary(predict_fn, X_train, Y_train, title="Neural Network")
    plt.show()

    #grad_check test
    parameters=initialize_parameters(X_train.shape[0],4,Y_train.shape[0])
    error_nh_4=gradient_checks_test(X_train,Y_train,parameters)

    accuracy_plot(history,parameters)
    plt.show()

    print("n_h=4 기울기 오차: ", error_nh_4)

    parameters=initialize_parameters(X_train.shape[0],50,Y_train.shape[0])
    error_nh_50=gradient_checks_test(X_train,Y_train,parameters)

    print("n_h=50 기울기 오차: ", error_nh_50)


    #experiment A

    def experiment_A():
        parameters_A_1,history_A_1=nn_model(X_train,Y_train,1,20000,0.2,X_test,Y_test)
        parameters_A_2,history_A_2=nn_model(X_train,Y_train,4,20000,0.2,X_test,Y_test)
        parameters_A_3,history_A_3=nn_model(X_train,Y_train,50,20000,0.2,X_test,Y_test)
        
        X_small, Y_small = make_data(40,   0.3, 7)
        X_test3, Y_test3 = make_data(1000, 0.3, 99)
        parameters_A_4,history_A_4=nn_model(X_small,Y_small,50,20000,0.2,X_test3,Y_test3)

        titles = ["A-1", "A-2", "A-3", "A-4"]
        params = [parameters_A_1, parameters_A_2, parameters_A_3, parameters_A_4]
        histories = [history_A_1, history_A_2, history_A_3, history_A_4]

        Xs = [X_train, X_train, X_train, X_small]
        Ys = [Y_train, Y_train, Y_train, Y_small]

        fig, axes = plt.subplots(1, 4, figsize=(12, 3))

        for i in range(4):
            plt.sca(axes[i])

            def predict_fn(X_grid):
                return predict(params[i], X_grid)

            plot_decision_boundary(predict_fn, Xs[i], Ys[i], titles[i])

        plt.tight_layout()
        plt.show()


        fig, axes = plt.subplots(1, 4, figsize=(12, 3))

        for i in range(4):
            plt.sca(axes[i])
            cost_function_plot(histories[i], params[i])
            plt.title(titles[i])

        plt.tight_layout()
        plt.show()

        train_accuracy = [
            history_A_1["train_accuracy"][-1],
            history_A_2["train_accuracy"][-1],
            history_A_3["train_accuracy"][-1],
            history_A_4["train_accuracy"][-1]
        ]

        test_accuracy = [
            history_A_1["test_accuracy"][-1],
            history_A_2["test_accuracy"][-1],
            history_A_3["test_accuracy"][-1],
            history_A_4["test_accuracy"][-1]
        ]

        train_cost = [
            history_A_1["cost_list"][-1],
            history_A_2["cost_list"][-1],
            history_A_3["cost_list"][-1],
            history_A_4["cost_list"][-1]
        ]

        test_cost = [
            history_A_1["test_cost_list"][-1],
            history_A_2["test_cost_list"][-1],
            history_A_3["test_cost_list"][-1],
            history_A_4["test_cost_list"][-1]
        ]

        min_test_cost = [
            np.min(history_A_1["test_cost_list"]),
            np.min(history_A_2["test_cost_list"]),
            np.min(history_A_3["test_cost_list"]),
            np.min(history_A_4["test_cost_list"])
        ]

        min_iteration = [
            np.argmin(history_A_1["test_cost_list"]) + 1,
            np.argmin(history_A_2["test_cost_list"]) + 1,
            np.argmin(history_A_3["test_cost_list"]) + 1,
            np.argmin(history_A_4["test_cost_list"]) + 1
        ]

        increase = [
            test_cost[0] - min_test_cost[0],
            test_cost[1] - min_test_cost[1],
            test_cost[2] - min_test_cost[2],
            test_cost[3] - min_test_cost[3]
        ]

        print("학습 정확도:", "A-1:", train_accuracy[0], "A-2:", train_accuracy[1], "A-3:", train_accuracy[2], "A-4:", train_accuracy[3])
        print("시험 정확도:", "A-1:", test_accuracy[0], "A-2:", test_accuracy[1], "A-3:", test_accuracy[2], "A-4:", test_accuracy[3])
        print("최종 학습 비용:", "A-1:", train_cost[0], "A-2:", train_cost[1], "A-3:", train_cost[2], "A-4:", train_cost[3])
        print("최종 시험 비용:", "A-1:", test_cost[0], "A-2:", test_cost[1], "A-3:", test_cost[2], "A-4:", test_cost[3])
        print("최소 시험 비용:", "A-1:", min_test_cost[0], "A-2:", min_test_cost[1], "A-3:", min_test_cost[2], "A-4:", min_test_cost[3])
        print("최소 비용 반복:", "A-1:", min_iteration[0], "A-2:", min_iteration[1], "A-3:", min_iteration[2], "A-4:", min_iteration[3])
        print("상승폭:", "A-1:", increase[0], "A-2:", increase[1], "A-3:", increase[2], "A-4:", increase[3])

    #experiment_A()

    # experiment B
    def experiment_B():
        def initialize_parameters(n_x, n_h, n_y):
            W1= np.zeros((n_h,n_x))
            b1=np.zeros((n_h,1))
            W2= np.zeros((n_y,n_h))
            b2=np.zeros((n_y,1))

            parameters={
                "W1":W1,
                "W2":W2,
                "b1":b1,
                "b2":b2
            }

            return parameters 
        parameters_B_1,history_B_1=nn_model(X_train,Y_train,50,20000,0.2,X_test,Y_test,initialize_parameters)

        def initialize_parameters(n_x, n_h, n_y):
            W1= np.zeros((n_h,n_x))
            b1=np.zeros((n_h,1))
            W2=np.random.randn(n_y, n_h)*0.5
            b2=np.zeros((n_y,1))

            parameters={
                "W1":W1,
                "W2":W2,
                "b1":b1,
                "b2":b2
            }

            return parameters
        parameters_B_2,history_B_2=nn_model(X_train,Y_train,50,20000,0.2,X_test,Y_test,initialize_parameters)

        def initialize_parameters(n_x, n_h, n_y):
            W1= np.random.randn(n_h, n_x)*0.5
            b1=np.zeros((n_h,1))
            W2= np.random.randn(n_y, n_h)*0.5
            b2=np.zeros((n_y,1))

            parameters={
                "W1":W1,
                "W2":W2,
                "b1":b1,
                "b2":b2
            }

            return parameters
        parameters_B_3,history_B_3=nn_model(X_train,Y_train,50,20000,0.2,X_test,Y_test,initialize_parameters)


        def initialize_parameters(n_x, n_h, n_y):
            W1= np.zeros((n_h,n_x)) +0.5
            b1=np.zeros((n_h,1))
            W2= np.zeros((n_y,n_h)) +0.5
            b2=np.zeros((n_y,1))

            parameters={
                "W1":W1,
                "W2":W2,
                "b1":b1,
                "b2":b2
            }

            return parameters 
        parameters_B_4,history_B_4=nn_model(X_train,Y_train,50,20000,0.2,X_test,Y_test,initialize_parameters)

        test_accuracy = [
            history_B_1["test_accuracy"][-1],
            history_B_2["test_accuracy"][-1],
            history_B_3["test_accuracy"][-1],
            history_B_4["test_accuracy"][-1]
        ]
        test_cost = [
        history_B_1["test_cost_list"][-1],
        history_B_2["test_cost_list"][-1],
        history_B_3["test_cost_list"][-1],
        history_B_4["test_cost_list"][-1]
        ]

        print("실험 B_1 W1 : ",np.unique(np.round(parameters_B_1["W1"], 8), axis=0).shape[0], "개의 서로 다른 은닉 유닛")
        print("실험 B_2 W1 : ",np.unique(np.round(parameters_B_2["W1"], 8), axis=0).shape[0], "개의 서로 다른 은닉 유닛")
        print("실험 B_3 W1 : ",np.unique(np.round(parameters_B_3["W1"], 8), axis=0).shape[0], "개의 서로 다른 은닉 유닛")
        print("실험 B_4 W1 : ",np.unique(np.round(parameters_B_4["W1"], 8), axis=0).shape[0], "개의 서로 다른 은닉 유닛")

        print("실험 B_1 W2 : ",np.unique(np.round(parameters_B_1["W2"], 8)).size, "개의 서로 다른 은닉 유닛")
        print("실험 B_2 W2 : ",np.unique(np.round(parameters_B_2["W2"], 8)).size, "개의 서로 다른 은닉 유닛")
        print("실험 B_3 W2 : ",np.unique(np.round(parameters_B_3["W2"], 8)).size    , "개의 서로 다른 은닉 유닛")
        print("실험 B_4 W2 : ",np.unique(np.round(parameters_B_4["W2"], 8)).size, "개의 서로 다른 은닉 유닛")

        print("시험 정확도:", "B-1:", test_accuracy[0], "B-2:", test_accuracy[1], "B-3:", test_accuracy[2], "B-4:", test_accuracy[3])
        histories = [history_B_1, history_B_2, history_B_3, history_B_4]
        params = [parameters_B_1, parameters_B_2, parameters_B_3, parameters_B_4]

        fig, axes = plt.subplots(1, 4, figsize=(12, 3))

        for i in range(4):
            plt.sca(axes[i])
            cost_function_plot(histories[i], params[i])
            plt.title(f"B-{i + 1}")

            if i == 0:
                plt.ylim(0.6, 0.8)
                plt.yticks([0.6, 0.65, 0.7, 0.75, 0.8],
                            ["0.60", "0.65", "0.70", "0.75", "0.80"])

        plt.tight_layout()
        plt.show()

    #experiment_B()
    
    #experiment C
    def experiment_C():
        parameters_C_4, history_C_4 = nn_model_C(
            X_train, Y_train, 4, 20000, 0.2, X_test, Y_test
        )

        parameters_C_50, history_C_50 = nn_model_C(
            X_train, Y_train, 50, 20000, 0.2, X_test, Y_test
        )

        print("C-1 시험 정확도:", history_C_4["test_accuracy"][-1])
        print("C-2 시험 정확도:", history_C_50["test_accuracy"][-1])
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))

        plt.sca(axes[0])

        def predict_fn(X_grid):
            return predict_C(parameters_C_4, X_grid)

        plot_decision_boundary(predict_fn, X_train, Y_train, title="C-1")


        plt.sca(axes[1])

        def predict_fn(X_grid):
            return predict_C(parameters_C_50, X_grid)

        plot_decision_boundary(predict_fn, X_train, Y_train, title="C-2")

        plt.tight_layout()
        plt.show()

    experiment_C()

    #experiment D
    
    def experiment_D():
        learning_rates = [0.5, 5, 20, 50]
        history_D = {}

        for learning_rate in learning_rates:
            w, b = initialize(X_train.shape[0])

            history = {
                "cost_list": [],
                "test_cost_list": []
            }

            for i in range(2000):
                grads, _ = logistic_propagate(
                    w, b, X_train, Y_train, X_train.shape[1]
                )

                w -= learning_rate * grads["dw"]
                b -= learning_rate * grads["db"]

                Z_train = w.T @ X_train + b
                Z_test = w.T @ X_test + b

                history["cost_list"].append(compute_cost(Z_train, Y_train))
                history["test_cost_list"].append(compute_cost(Z_test, Y_test))

            history_D[learning_rate] = history
            history["w"] = w.copy()
            history["b"] = float(b)
            history_D[learning_rate] = history

        plt.figure(figsize=(8, 4))

        for learning_rate in learning_rates:
            plt.plot(
                np.arange(1, 101),
                history_D[learning_rate]["cost_list"][:100],
                label=f"lr={learning_rate}"
            )

        plt.xlabel("num_iterations")
        plt.ylabel("train cost")
        plt.title("Experiment D")
        plt.legend()
        plt.tight_layout()
        plt.show()

        for learning_rate in learning_rates:
            history = history_D[learning_rate]

            H, L, limit = logistic_hessian(
                X_train, history["w"], history["b"]
            )

            history["hessian"] = H
            history["L"] = L
            history["2/L"] = limit

            print(f"학습률: {learning_rate}, L: {L:.6f}, 2/L: {limit:.6f}")

        return history_D




    history_D = experiment_D()


if __name__ == "__main__":
    main()