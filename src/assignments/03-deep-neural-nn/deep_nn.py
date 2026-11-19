import numpy as np
np.random.seed(42)


import matplotlib.pyplot as plt
import sys

sys.path.insert(
    0,
    r"../02-shallow-neural-network"
)

from shallow_nn import plot_decision_boundary
sys.path.insert(
    0,
    r"../01-logistic-regression"
)
from logistic_regression import initialize, optimize, predict as logistic_predict   
from logistic_regression import propagate as logistic_propagate

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
def initialize_parameters_deep(layer_dims,seed=None):
    parameters = {}

    if (seed != None):
        np.random.seed(seed)

    for l in range(1,len(layer_dims)):

        parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2.0 / layer_dims[l-1])
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))

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
        dZ = dA * (cache["Z"] > 0)
    if  (activation == sigmoid):
        dZ=dA* cache["Z"]*(1-cache["Z"])

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
def L_layer_model(X, Y, layer_dims, num_iterations, learning_rate, X_test, Y_test,parameters=None):
    L=len(layer_dims)
    if parameters is None:
        parameters = initialize_parameters_deep(layer_dims)
    else:
          parameters = {
              key: value.copy()
              for key, value in parameters.items()
          }

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

#평탄화 
def flatten_parame(parameters):
    L = len(parameters) // 2
    parts = []

    for i in range(1, L + 1):
        W_reshape = np.asarray(parameters[f"W{i}"]).flatten()
        b_reshape = np.asarray(parameters[f"b{i}"]).flatten()

        parts.append(W_reshape)
        parts.append(b_reshape)

    theta = np.concatenate(parts)
    return theta

def flatten_grads(grads):
    L = sum(key.startswith("dW") for key in grads)
    parts = []

    for i in range(1, L+1 ):
        dW_reshape = np.asarray(grads[f"dW{i}"]).flatten()
        db_reshape = np.asarray(grads[f"db{i}"]).flatten()

        parts.append(dW_reshape)
        parts.append(db_reshape)

    theta = np.concatenate(parts)
    return theta

#기울기 검증 
from utils.grad_check import grad_check
def gradient_checks_test(X,Y):
    #초기 파라미터 설정
    layer_dims = [2, 4, 3, 1]
    L=len(layer_dims)
    parameters={}
    for l in range(1,L):

        parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2.0 / layer_dims[l-1])
        parameters[f"b{l}"] = np.random.randn(layer_dims[l], 1) * np.sqrt(2.0 / layer_dims[l-1])
    
    #데이터 준비


    indices = np.random.permutation(X.shape[1])[:20]
    X_shuffled = X[:, indices]
    Y_shuffled = Y[:, indices]

    #기울기 계산 
    AL,ZL,caches=L_model_forward(X_shuffled,parameters)
    grads=L_model_backward(ZL,Y_shuffled,caches)

    analytic_grad = flatten_grads(grads)

    #theta용 비용함수
    def theta_cost_function(theta):
        restored_parameters = {}
        start = 0

        for l in range(1, L):
            for key in (f"W{l}", f"b{l}"):
                size = parameters[key].size
                shape = parameters[key].shape

                restored_parameters[key] = theta[start:start + size].reshape(shape)
                start += size

        _, ZL, _ = L_model_forward(X_shuffled, restored_parameters)
        return compute_cost(ZL, Y_shuffled)
    
    #실제 기울기 검증 
    theta=flatten_parame(parameters)
    error=grad_check(theta_cost_function,theta,analytic_grad,epsilon=1e-5)

    return error

#비용함수 곡선 
def cost_function_plot(history):
    horizontal = np.arange(1, len(history["cost_list"]) + 1)
    plt.plot(horizontal, history["cost_list"], label="train")
    plt.plot(horizontal, history["test_cost_list"], label="test")

    plt.title(f"cost_function")
    plt.xlabel("num_iterations")
    plt.ylabel("cost")
    plt.yscale("log")
    plt.legend()

#학습 곡선만 그림
def train_cost_function_plot(history):
    horizontal = np.arange(1, len(history["cost_list"]) + 1)
    plt.plot(horizontal, history["cost_list"], label="train")



#정확도 곡선
def accuracy_plot(history):
    horizontal = np.arange(1, len(history["cost_list"]) + 1)
    plt.plot(history["accuracy_iterations"], history["train_accuracy"], label="train")
    plt.plot(history["accuracy_iterations"], history["test_accuracy"], label="test")

    plt.title(f"n_h={n_h}")
    plt.xlabel("num_iterations")
    plt.ylabel("accuracy")
    plt.legend()

#실험 A

def count_params(dims):
    return sum(dims[l]*dims[l-1] + dims[l] for l in range(1, len(dims)))

def historys_to_test(historys):
    L=len(historys)
    min_iteration=[]
    train_accuracy=[]
    test_accuracy=[]
    first_iteration_array=[]
    

    for i in range(L):
        min_iteration.append(np.argmin(historys[i]["test_cost_list"]) +1)
        train_accuracy.append(historys[i]["train_accuracy"][-1])
        test_accuracy.append(historys[i]["test_accuracy"][-1])

    for i in range(L):  
        accuracy = np.asarray(historys[i]["test_accuracy"])
        indices = np.flatnonzero(accuracy >= 0.9)
        if indices.size > 0:
            first_iteration = historys[i]["accuracy_iterations"][indices[0]]
        else:
            first_iteration = "-"
        first_iteration_array.append(first_iteration)
    

    return min_iteration,train_accuracy,test_accuracy,first_iteration_array


def experiment_A(X,Y,X_test,Y_test):
    A_layer_dim={}
    A_layer_dim["A-1"]= [2, 1]
    A_layer_dim["A-2"]= [2, 16, 1]
    A_layer_dim["A-3"]= [2, 64, 1]
    A_layer_dim["A-4"]= [2, 256, 1]
    A_layer_dim["A-5"]= [2, 16, 16, 1]   
    A_layer_dim["A-6"]= [2, 32, 16, 8, 1]

    L=len(A_layer_dim)
    count_param=[]
    historys=[]
    
    plt.figure(figsize=(18, 5))
    for i in range(1,L+1):
        count_param.append(count_params(A_layer_dim[f"A-{i}"]))
        parameters,history=L_layer_model(X, Y, A_layer_dim[f"A-{i}"], 30000, 0.2, X_test, Y_test)
        historys.append(history)
        

        if i in (1, 5, 6):

            plt.subplot(1, 3, (1, 5, 6).index(i) + 1)

            plot_decision_boundary(
                lambda X_grid: predict(parameters, X_grid),
                X, Y,
                title=f"A-{i}: {A_layer_dim[f'A-{i}']}"
            )

            plt.xlabel("x1")
            plt.ylabel("x2")

    min_iteration,train_accuracy,test_accuracy,first_iteration_array = historys_to_test(historys)

    total_result={
        "A_layer_dim":A_layer_dim,
        "count_param":count_param,
        "min_iteration":min_iteration,
        "train_accuracy":train_accuracy,
        "test_accuracy":test_accuracy,
        "first_iteration_array":first_iteration_array
    }

    return total_result
    
        
def experiment_A_extra2(X,Y,X_test,Y_test):
    A_layer_dim={}

    A_layer_dim["A-3"]= [2, 64, 1]
    A_layer_dim["A-5"]= [2, 16, 16, 1]   


    L=len(A_layer_dim)
    count_param=[]
    historys_A_3=[]
    historys_A_5=[]
    parameters_seeds={}
    for i in range(1,4):
        parameters_seeds[f"A-3_seed={i}"]=initialize_parameters_deep(A_layer_dim["A-3"],i)
        parameters_seeds[f"A-5_seed={i}"]=initialize_parameters_deep(A_layer_dim["A-5"],i)

    for i in range(1,4):
        parameters,history_A_3=L_layer_model(X, Y, A_layer_dim[f"A-{3}"], 30000, 0.2, X_test, Y_test,parameters_seeds[f"A-3_seed={i}"])
        parameters,history_A_5=L_layer_model(X, Y, A_layer_dim[f"A-{5}"], 30000, 0.2, X_test, Y_test,parameters_seeds[f"A-5_seed={i}"])
        historys_A_3.append(history_A_3)
        historys_A_5.append(history_A_5)

    min_iteration_A_3,train_accuracy_A_3,test_accuracy_A_3,first_iteration_array_A_3 = historys_to_test(historys_A_3)
    min_iteration_A_5,train_accuracy_A_5,test_accuracy_A_5,first_iteration_array_A_5 = historys_to_test(historys_A_5)

    for i in range(0,3):
        print(f"A-3 seed={i} 시험 정확도",test_accuracy_A_3[i])
        print(f"A-5 seed={i} 시험 정확도",test_accuracy_A_5[i])

    return None

def experiment_B(X,Y,X_test,Y_test):
    B_layer_dim={}
    B_layer_dim["B-1"]= [2, 16, 1]
    B_layer_dim["B-2"]= [2, 16, 16, 1]
    B_layer_dim["B-3"]= [2, 32, 16, 8, 1]	

    L=len(B_layer_dim)
    historys=[]
    

    for i in range(1,L+1):
        parameters,history=L_layer_model(X, Y, B_layer_dim[f"B-{i}"], 30000, 0.2, X_test, Y_test)
        historys.append(history)

    _,_,test_accuracy_He,_ = historys_to_test(historys)


    parameters_list=[]

    historys_randn=[]



    for i in range(1,L+1):
        parameters = {}
        layer_dims=B_layer_dim[f"B-{i}"]
        for l in range(1,len(layer_dims)):

            parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
            parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))
        parameters_list.append(parameters)
    
    for i in range(1,L+1):
        parameters,history_randn=L_layer_model(X, Y, B_layer_dim[f"B-{i}"], 30000, 0.2, X_test, Y_test,parameters_list[i-1])
        historys_randn.append(history_randn)

    _,_,test_accuracy_randn,_ = historys_to_test(historys_randn)

    for i in range(L):
        horizontal = np.arange(1, len(history["cost_list"]) + 1)
        plt.plot(horizontal, historys[i]["cost_list"], label=f"He")
        plt.plot(horizontal, historys_randn[i]["cost_list"], label=f"randn")
        plt.title(f"B-{i+1}")
        plt.xlabel("num_iterations")
        plt.ylabel("cost")
        plt.yscale("log")
        plt.legend()
        plt.show()

    return test_accuracy_He,test_accuracy_randn


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
    print("훈련 정확도: ", history["train_accuracy"][-1])
    print("시험 정확도: ",history["test_accuracy"][-1])

#기울기 검증 테스트 
    error=gradient_checks_test(X_train,Y_train)
    print("기울기 오차: ", error)

#환원 검증 
    layer_dims = [2, 1]

    parameters = {}
    for l in range(1,len(layer_dims)):

        parameters[f"W{l}"] = np.zeros((layer_dims[l],layer_dims[l-1]))
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))


    parameters,history=L_layer_model(X_train, Y_train, layer_dims, 30000, 0.2, X_test, Y_test,parameters)
    print("환원 훈련 정확도: ", history["train_accuracy"][-1])
    print("환원 정확도: ",history["test_accuracy"][-1])

    def predict_fn(X_grid):
        w, b = initialize(X_train.shape[0])

        params, _, _ = optimize(
            w, b, X_train, Y_train,
            num_iterations=30000,
            learning_rate=0.2
        )

        return logistic_predict(params["w"], params["b"], X_grid)

    predict_train=predict_fn(X_train)
    train_accuracy=np.mean(predict_train == Y_train)
    print("로지스틱 회귀 훈련 정확도: ",train_accuracy)
    
    predict_test=predict_fn(X_test)
    test_accuracy=np.mean(predict_test == Y_test)
    print("로지스틱 회귀 시험 정확도: ",test_accuracy)

    # 시험 정확도 체크
    layer_dims = [2, 16, 16, 1]
    parameters,history=L_layer_model(X_train, Y_train, layer_dims, 30000, 0.2, X_test, Y_test)
    print("훈련 정확도: ", history["train_accuracy"][-1],"시험 정확도: ",history["test_accuracy"][-1])

    cost_function_plot(history)
    plt.show()

    #실험 A
    def run_experiment_A_result():
        total_result=experiment_A(X_train,Y_train,X_test,Y_test)


        for i in range(6):
            name = f"A-{i + 1}"
            layer_dims = total_result["A_layer_dim"][name]


            print(name ,"레이어 수: ",total_result["A_layer_dim"][name],"파리미터 수: ",total_result["count_param"][i] ,
                "훈련 정화도: ",total_result["train_accuracy"][i],"시험 정확도: ",total_result["test_accuracy"][i], 
                "시험 최소 비용: ",total_result["min_iteration"][i],"시험 정확도 90% 도달",total_result["first_iteration_array"][i])


    def run_experiment_A_extra1(): 
        A_layer_dim={}
        A_layer_dim["A-4"]= [2, 256, 1]

        parameters_seeds=initialize_parameters_deep(A_layer_dim["A-4"],1)
        parameters,history=L_layer_model(X_train, Y_train, A_layer_dim[f"A-{4}"], 100000, 0.2, X_test, Y_test)
        min_iteration,train_accuracy,test_accuracy,first_iteration_array=historys_to_test([history])
        print("A-4 추가 실험 " ,"레이어 수: ",A_layer_dim["A-4"],"파리미터 수: ",count_params(A_layer_dim["A-4"]) ,
                    "훈련 정화도: ",train_accuracy[0],"시험 정확도: ",test_accuracy[0], 
                    "시험 최소 비용: ",min_iteration[0],"시험 정확도 90% 도달",first_iteration_array[0])
        

    def run_experiment_A_extra2(): 
        experiment_A_extra2(X_train,Y_train,X_test,Y_test)

    
    run_experiment_A_result()
    plt.show()
    run_experiment_A_extra1()
    run_experiment_A_extra2()


    def run_experiment_B():
        test_accuracy_He,test_accuracy_randn=experiment_B(X_train,Y_train,X_test,Y_test)

        for i in range(3):
            name = f"B-{i + 1}"
            print(name,"He 시험 정확도: ", test_accuracy_He[i],"randn 시험 정확도: ", test_accuracy_randn[i])

    run_experiment_B()

    
if __name__ == "__main__":
    main()
