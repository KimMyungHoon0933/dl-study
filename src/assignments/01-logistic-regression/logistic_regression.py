# 0. 데이터 셋 준비
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils.grad_check import grad_check

digits = load_digits()
mask = (digits.target == 0) | (digits.target == 1)
X, y = digits.data[mask], digits.target[mask]        # (360, 64), (360,)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)

X_train = X_tr.T / 16.0        # (64, 288)
X_test  = X_te.T / 16.0        # (64, 72)
Y_train = y_tr.reshape(1, -1)  # (1, 288)
Y_test  = y_te.reshape(1, -1)  # (1, 72)


#print(X_train.shape)



#클래스 0 이미지 
label_0 =digits.target==0
data_label_0=digits.data[label_0]

#클래스 1 이미지 
label_1 =digits.target==1
data_label_1=digits.data[label_1]
fig ,axes = plt.subplots(1,5,figsize=(10,2))

for i, ax in enumerate(axes):
    ax.imshow(data_label_0[i].reshape(8,8),cmap='gray')
    ax.set_title("label 0")
    ax.axis("off")
plt.tight_layout()
#plt.show()

fig ,axes = plt.subplots(1,5,figsize=(10,2))

for i, ax in enumerate(axes):
    ax.imshow(data_label_1[i,].reshape(8,8),cmap='gray')
    ax.set_title("label 1")
    ax.axis("off")
plt.tight_layout()
#plt.show()

# sigmoid
def sigmoid(z):
    #예상 입력: z=(1,m)
    return np.exp(-np.logaddexp(0, -z)) #출력 z.shape=(1,m)

overflow_debug=sigmoid(np.array([-1000, 0, 1000]))
print(overflow_debug)

#initialize
def initialize(dim):
    w=np.zeros((dim,1))
    b=np.zeros(1)
    return w,b #w.shape=(dim,1) b=(1,)

#w,b= initialize(64)
#print("w shape:",w.shape,"b shape: ",b.shape)

#propagate
def cost_function(a,Y,m):
    #a.shape=(1,m) Y.shape=(1,m) 
    a[a == 0]=1e-15 
    a[a == 1] = 1 - 1e-7    
    L=-(Y*np.log(a)+(1-Y)* np.log(1-a))
    J=np.sum(L)/m
    return J #j.shape=()

def propagate(w,b,X,Y,m):
    #w.shape=(64,1) b.shape(1,) X_train.shape=(64,288) Y_train.shape=(1,288) m.shape=()
    Z=w.T@ X + b
    A=sigmoid(Z)
    cost=cost_function(A,Y,m)
    dZ= A-Y 
    dw= (X @ dZ.T)/m
    db= np.mean(dZ)
    grads = {
    "dw": dw,
    "db": db
    }
    return grads,cost #grads["dw"].shape=(64,1) grads["db"].shape=(1,),cost.shape=()

#w,b= initialize(64)
#grads,cost=propagate(w,b,X_train,Y_train,288)
#print(grads["dw"],grads["db"],cost.shape)


#Y_debug = np.array([[0, 1]])

#A_debug = sigmoid(Z_debug)

#cost_debug = cost_function(
#        A_debug, Y_debug, Y_debug.shape[1]
#)
            
#print("cost:", cost_debug)
#print("NaN 포함:", np.isnan(cost_debug).any())
#print("Inf 포함:", np.isinf(cost_debug).any())


#optimize
def optimize(w, b, X, Y, num_iterations, learning_rate,grad_check_bool=False):
    #w.shape=(64,1) b.shape(1,) X_train.shape=(64,288) Y_train.shape=(1,288) 
    cost_array=[]
    grads_list=[]
    for i in range(num_iterations):
        grads,cost=propagate(w,b,X,Y,X.shape[1])

        if(grad_check_bool):
            theta = flatten(w, b)
            analytic_grad = flatten(grads["dw"], grads["db"])

            error=grad_check(theta_cost_function,theta,analytic_grad)
            print(error)
        
        w -= learning_rate * grads["dw"]
        b -= learning_rate * grads["db"]
        
        if((i%100)==0):
            cost_array.append(cost)
            grads_list.append(grads)
    
    params= {
    "w": w,
    "b": b
    }
    grads_list = np.array(grads_list)
    cost_array=np.array(cost_array)

    #출력: params["w"].shape=(64,1) params["b"].shape=(1,),grads_list.shape=(20,),cost_array.shape=(20,)
    return params,grads_list,cost_array 

#params , grads ,cost= optimize(w,b,X_train,Y_train,1000,0.05)

#print(params["w"].shape,grads.shape,cost.shape)

#predict
def predict(w,b,X):
    #w.shape=(64,1) b.shape(1,) 
    # x.shape= X_train(64,m) 
    Z=w.T@X+b
    A=sigmoid(Z)
    y_pred=(A >= 0.5).astype(int)
    return y_pred #y_pred.shpae(m,)

#y_pred=predict(params["w"],params["b"],X_test)
#print(y_pred.shape)  
#print(y_pred)     

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.05,grad_check_bool=False):
    #w.shape=(64,1) b.shape(1,) X_train.shape=(64,288) Y_train.shape=(1,288) 
    w,b=initialize(X_train.shape[0])

    params,grads_list,cost_list = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,grad_check_bool)

    w=params["w"]
    b=params["b"]

    train_pred=predict(w,b,X_train)
    test_pred=predict(w,b,X_test)

    train_accuracy=np.mean(train_pred == Y_train)
    test_accuracy=np.mean(test_pred == Y_test)

    return {
        "w": w,
        "b": b,
        "grads_list":grads_list,
        "cost_list": cost_list,
        "train_accuracy": np.mean(train_pred == Y_train),
        "test_accuracy": np.mean(test_pred == Y_test),
    } #w.shape=(64,1) b=(1,) grads_list.shape=(20,),cost_array.shape=(20,)
      #train_accuracy.shape=(1,) test_accuracy.shape=(1,)

#result = model(X_train, Y_train, X_test, Y_test)

#print(f"훈련 정확도: {result['train_accuracy']:.2%}")
#print(f"테스트 정확도: {result['test_accuracy']:.2%}")

def flatten(w,b):
    w_reshape=np.asarray(w).flatten()
    b_reshape=np.asarray(b).flatten()
    theta=np.concat([w_reshape,b_reshape])
    return theta

def theta_cost_function(theta):
    w=theta[:-1].reshape(-1,1)
    b=theta[-1:]
    Z=w.T@X_check+b 
    a=sigmoid(Z)
    a[a == 0]=1e-7
    a[a == 1] = 1 - 1e-7    
    L=-(Y_check*np.log(a)+(1-Y_check)* np.log(1-a))
    J=np.sum(L)/X_check.shape[1]
    
     


    return J #j.shape=()





X_check = X_train[:, :10]
Y_check = Y_train[:, :10]

result = model(X_check, Y_check, X_check, Y_check,grad_check_bool=True,num_iterations=10)

result = model(X_train, Y_train, X_test, Y_test)

print(f"훈련 정확도: {result['train_accuracy']:.2%}")
print(f"테스트 정확도: {result['test_accuracy']:.2%}")

#grads, _ = propagate(
#    w, b, X_check, Y_check, X_check.shape[1]
#)

#analytic_grad = flatten(grads["dw"], grads["db"])

#theta=flatten(w,b)
#error=grad_check(theta_cost_function,theta,analytic_grad)

#print(error)





    