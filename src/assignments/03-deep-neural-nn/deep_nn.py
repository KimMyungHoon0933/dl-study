import numpy as np
np.random.seed(42)


import matplotlib.pyplot as plt
import sys

sys.path.insert(
    0,
    r"../02-shallow-neural-network"
)

from shallow_nn import plot_decision_boundary

#활성화 함수
def sigmoid(z):
    #z=(4,n)
    z=np.array(z)
    A=np.zeros_like(z, dtype=float)
    positive = (z >= 0)
    
    A[positive]=1 / (1 + np.exp(-z[positive]))

    exp_z = np.exp(z[~positive])
    A[~positive] = exp_z / (1 + exp_z)

    return A # A = (1,n)

def relu(x):
    x=np.maximum(0,x)
    return x 


#데이터 
def make_spiral(n_per_class=200, noise=0.12, seed=0, turns=1.6):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k in (0, 1):
        r = np.linspace(0.08, 1.0, n_per_class)
        t = np.linspace(0, turns*2*np.pi, n_per_class) + k*np.pi
        xx = r*np.sin(t) + rng.normal(scale=noise, size=n_per_class)*r
        yy = r*np.cos(t) + rng.normal(scale=noise, size=n_per_class)*r
        X.append(np.stack([xx, yy], 1)); y.append(np.full(n_per_class, k))
    return np.concatenate(X).T, np.concatenate(y).reshape(1, -1)


#가중치 초기화
def initialize_parameters_deep(layer_dims):
    parameters = {}

    for l in range(1,len(layer_dims)):

        parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2.0 / layer_dims[l-1])
        parameters[f"b{l}"] = np.random.randn(layer_dims[l], 1) * np.sqrt(2.0 / layer_dims[l-1])

    return parameters

#선형 결합
def linear_forward(A_prev, W, b):
    cache={}
    Z = W @ A_prev + b
    cache["W"] = W 
    cache["b"] = b 
    cache["Z"] = Z 
    cache["A_prev"]=A_prev
    return Z,cache
    
def linear_activation_forward(A_prev, W, b, activation):

    Z,cache = linear_forward(A_prev, W, b)
    A = activation(Z)

    cache["A"]=A
    cache["activation"]=activation

    return A,cache

#순전파
def L_model_forward(X, parameters):
    caches={}
    A_prev=X
    L = len(parameters) // 2
    for l in range(1,L+1):

        if (l==L):
            A,cache= linear_activation_forward(A_prev,parameters[f"W{l}"],parameters[f"b{l}"],sigmoid)
            AL=A
            ZL=cache["Z"]
        else:
            A,cache= linear_activation_forward(A_prev,parameters[f"W{l}"],parameters[f"b{l}"],relu)

        A_prev=A
        caches[f"{l}"]=cache


    return AL,ZL,caches

#비용함수
def compute_cost(ZL,Y):
    L = np.logaddexp(0, ZL) - Y * ZL
    J=np.sum(L)/ZL.shape[1]
    return J #(1,n)

# 역전파 파라미터
def linear_backward(dZ, cache):
    dW= (dZ @ cache["A_prev"].T) /dZ.shape[1]
    db = np.sum(dZ,axis=1 ,keepdims= True)/dZ.shape[1]
    dA_prev= cache["W"].T @dZ
    return dA_prev,dW,db

# 역전파 dz 계산
def linear_activation_backward(dA, cache, activation):
    if (activation == relu):
        dZ = dA * (cache["A"] > 0)
    if  (activation == sigmoid):
        dZ=dA* cache["A"]*(1-cache["A"])

    dA_prev,dW,db=linear_backward(dZ, cache)
    return dA_prev,dW,db
        
#모델 역전파 
def L_model_backward(ZL, Y, caches):
    grads={}
    L= len(caches)

    dZ = sigmoid(ZL) - Y
    dA_prev, dW, db = linear_backward(dZ, caches[f"{L}"])

    grads[f"dA{L-1}"] = dA_prev
    grads[f"dW{L}"] = dW
    grads[f"db{L}"] = db

    for i in range(L-1, 0, -1):
        cache=caches[f"{i}"]
        activation=cache["activation"]
        dA_prev ,dW,db=linear_activation_backward(dA_prev,cache,activation)
        grads[f"dA{i-1}"]=dA_prev
        grads[f"dW{i}"]=dW
        grads[f"db{i}"]=db

    
    return grads

#가중치 업데이트
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    
    for i in range (1,L+1):
        parameters[f"W{i}"]=  parameters[f"W{i}"] - learning_rate * grads[f"dW{i}"]
        parameters[f"b{i}"]=  parameters[f"b{i}"] - learning_rate * grads[f"db{i}"]

    return parameters

#예측
def predict(parameters, X):
    AL,_,_=L_model_forward(X,parameters)
    predict= (AL >= 0.5).astype(int)
    return predict

#전체 모델 
def L_layer_model(X, Y, layer_dims, num_iterations, learning_rate, X_test, Y_test):
    L=len(layer_dims)
    parameters=initialize_parameters_deep(layer_dims)
    history = {
        "cost_list": [],
        "test_cost_list": [],
        "train_accuracy": [],
        "test_accuracy": [],
        "accuracy_iterations":[]
    }

    for i in range(num_iterations):
        AL,ZL,caches=L_model_forward(X, parameters)
        grads=L_model_backward(ZL, Y, caches)
        parameters=update_parameters(parameters,grads,learning_rate)


        #비용 계산 
        AL,ZL,caches_cost_train=L_model_forward(X, parameters)
        AL,ZL,caches_cost_test=L_model_forward(X_test, parameters)

        append_cost_function=compute_cost(caches_cost_train[f"{L-1}"]["Z"], Y)
        test_cost_function=compute_cost(caches_cost_test[f"{L-1}"]["Z"], Y_test)

        history["cost_list"].append(append_cost_function)
        history["test_cost_list"].append(
            test_cost_function
        )

        if((i+1)%100==0):
            predict_train = predict(parameters, X)
            predict_test = predict(parameters, X_test)

            history["accuracy_iterations"].append(i + 1)
            history["train_accuracy"].append(np.mean(predict_train == Y))
            history["test_accuracy"].append(np.mean(predict_test == Y_test))

    return parameters, history






def main():
    X_train, Y_train = make_spiral(200, 0.12, 0)    # (2, 400)
    X_test,  Y_test  = make_spiral(500, 0.12, 99)   # (2, 1000)

    print(X_train.shape,X_test.shape)
    print(Y_train.shape,Y_test.shape)

#    print(Y_test[0:10])
#  데이터 출력
#    plt.scatter(X_train[0], X_train[1], c=Y_train.ravel(), cmap="coolwarm", s=15)
#    plt.title("train")
#    plt.tight_layout()
#    plt.show()

#    plt.scatter(X_test[0], X_test[1], c=Y_test.ravel(), cmap="coolwarm", s=15)
#    plt.title("test")
#    plt.tight_layout()
#    plt.show()

#  초기화 함수 테스트
    layer_dims=[2, 16, 16, 1]
    parameters=initialize_parameters_deep(layer_dims)
    print(parameters["W1"].shape,parameters["W2"].shape,parameters["W3"].shape)
    print(parameters["b1"].shape,parameters["b2"].shape,parameters["b3"].shape)

# 선형 결함 함수 테스트 
    Z1,cache=linear_forward(X_train[:,0],parameters["W1"],parameters["b1"])
    print(Z1.shape)

# 선형 결함 함수 테스트 
    A1,cache=linear_activation_forward(X_train[:,0],parameters["W1"],parameters["b1"],relu)
    print(A1.shape)

# 모델 순전파 테스트    
    AL,ZL,caches=L_model_forward(X_train,parameters)
    print(AL.shape,ZL.shape)
    print(len(caches))

# 비용함수 테스트 
    J=compute_cost(ZL,Y_train)
    print(J.shape)

#역전파 테스트 
    grads=L_model_backward(ZL,Y_train,caches)
    print(grads["dW1"].shape,grads["dW2"].shape,grads["dW3"].shape)

# 가중치 업데이트 테스트 
    parameters=update_parameters(parameters,grads,0.2)
    print(parameters["W1"].shape,parameters["W2"].shape,parameters["W3"].shape)
    print(parameters["b1"].shape,parameters["b2"].shape,parameters["b3"].shape)

#전체 모델 테스트 
    parameters,history=L_layer_model(X_train, Y_train, layer_dims, 30000, 0.2, X_test, Y_test)
    print(parameters["W1"].shape,parameters["W2"].shape,parameters["W3"].shape)

if __name__ == "__main__":
    main()
