import numpy as np
np.random.seed(42)

# 0. 데이터 셋 준비

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


values, counts = np.unique(Y_train, return_counts=True)
print("X_train.shape: ",X_train.shape ,"X_train의 최솟값: ",X_train.min() , " X_train_최대값: ", X_train.max())
print("Y_train의 값 종류: ",np.unique(Y_train))



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
plt.show()

fig ,axes = plt.subplots(1,5,figsize=(10,2))

for i, ax in enumerate(axes):
    ax.imshow(data_label_1[i,].reshape(8,8),cmap='gray')
    ax.set_title("label 1")
    ax.axis("off")
plt.tight_layout()
plt.show()

# sigmoid
def sigmoid(z):
    #예상 입력: z=(1,m)
    return np.exp(-np.logaddexp(0, -z)) #출력 z.shape=(1,m)

overflow_debug=sigmoid(np.array([-1000, 0, 1000]))
print(overflow_debug)

#initialize
def initialize(dim):
    w=np.zeros((dim,1))
    b=0.0
    return w,b #w.shape=(dim,1) b=()

w,b= initialize(64)
print("w shape:",w.shape)

#propagate
def cost_function(a,Y,m):
    #a.shape=(1,m) Y.shape=(1,m) 
    a_temp = a.copy()
    a_temp[a_temp == 0]=1e-15 
    a_temp[a_temp == 1] = 1 - 1e-7    
    L=-(Y*np.log(a_temp)+(1-Y)* np.log(1-a_temp))
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
    return grads,cost #grads["dw"].shape=(64,1) grads["db"].shape=(),cost.shape=()

w,b= initialize(64)
grads,cost=propagate(w,b,X_train,Y_train,288)
print(grads["dw"],grads["db"],cost.shape)


Y_debug = np.array([[0, 1]])

A_debug = sigmoid(Y_debug)

cost_debug = cost_function(
        A_debug, Y_debug, Y_debug.shape[1]
)
            
print("cost:", cost_debug)
print("NaN 포함:", np.isnan(cost_debug).any())
print("Inf 포함:", np.isinf(cost_debug).any())


#optimize
def optimize(w, b, X, Y, num_iterations, learning_rate):
    #w.shape=(64,1) b.shape(1,) X_train.shape=(64,288) Y_train.shape=(1,288) 
    cost_array=[]
    grads_list=[]
    for i in range(num_iterations):
        grads,cost=propagate(w,b,X,Y,X.shape[1])

        #if(grad_check_bool):
        #    theta = flatten(w, b)
        #    analytic_grad = flatten(grads["dw"], grads["db"])

        #    error=grad_check(theta_cost_function,theta,analytic_grad)
        #    print(error)
        
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

params , grads ,cost= optimize(w,b,X_train,Y_train,1000,0.05)

print(params["w"].shape,grads.shape,cost.shape)

#predict
def predict(w,b,X):
    #w.shape=(64,1) b.shape(1,) 
    # x.shape= X_train(64,m) 
    Z=w.T@X+b
    A=sigmoid(Z)
    y_pred=(A >= 0.5).astype(int)
    return y_pred #y_pred.shpae(1,m)

y_pred=predict(params["w"],params["b"],X_test)
print(y_pred.shape)  
print(y_pred)     

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.05):
    #w.shape=(64,1) b.shape(1,) X_train.shape=(64,288) Y_train.shape=(1,288) 
    w,b=initialize(X_train.shape[0])

    params,grads_list,cost_list = optimize(w,b,X_train,Y_train,num_iterations,learning_rate)

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

result = model(X_train, Y_train, X_test, Y_test)

print(f"훈련 정확도: {result['train_accuracy']:.2%}")
print(f"테스트 정확도: {result['test_accuracy']:.2%}")

def flatten(w,b):
    w_reshape=np.asarray(w).flatten()
    b_reshape=np.asarray(b).flatten()
    theta=np.concat([w_reshape,b_reshape])
    return theta


    
     


    return J #j.shape=()

def gradient_checks_test(X,Y,w,b):
    X_check = X[:, :10]
    Y_check = Y[:, :10]

    def theta_cost_function(theta):
        w=theta[:-1].reshape(-1,1)
        b=theta[-1:]
        Z=w.T@X_check+b 
        A=sigmoid(Z)
        return cost_function(A, Y_check, X_check.shape[1])
    
    grads, _ = propagate(
        w, b, X_check, Y_check, X_check.shape[1]
    )

    analytic_grad = flatten(grads["dw"], grads["db"])

    theta=flatten(w,b)
    error=grad_check(theta_cost_function,theta,analytic_grad)

    return error

w,b=initialize(X_train.shape[0])
print(gradient_checks_test(X_train,Y_train,w,b))

result = model(X_train, Y_train, X_test, Y_test)

print(f"훈련 정확도: {result['train_accuracy']:.2%}")
print(f"테스트 정확도: {result['test_accuracy']:.2%}")


horizontal  = np.linspace(0, 1900, 20)
plt.plot(horizontal,result["cost_list"])
plt.xlabel("num_iterations")
plt.ylabel("cost_function")
plt.show()

print(result["cost_list"].shape)

#추가 실험

mask_3_5 = (digits.target == 3) | (digits.target == 5)
X_3_5, y_3_5 = digits.data[mask_3_5], digits.target[mask_3_5]        # (360, 64), (360,)

for i in range(y_3_5.shape[0]):
    if (y_3_5[i]==3):
        y_3_5[i]=0
    elif(y_3_5[i]==5):
        y_3_5[i]=1


X_tr_3_5, X_te_3_5, y_tr_3_5, y_te_3_5 = train_test_split(
    X_3_5, y_3_5, test_size=0.2, random_state=0, stratify=y_3_5)

X_train_3_5 = X_tr_3_5.T / 16.0        # (64, 288)
X_test_3_5  = X_te_3_5.T / 16.0        # (64, 72)
Y_train_3_5 = y_tr_3_5.reshape(1, -1)  # (1, 288)
Y_test_3_5  = y_te_3_5.reshape(1, -1)  # (1, 72)

result_3_5 = model(
    X_train_3_5, Y_train_3_5,
    X_test_3_5, Y_test_3_5
)

print(f"3 5훈련 정확도: {result_3_5['train_accuracy']:.2%}")
print(f"3 5테스트 정확도: {result_3_5['test_accuracy']:.2%}")

w = result_3_5["w"]
b = result_3_5["b"]

y_train_pred_3_5=np.array(predict(w,b,X_train_3_5))
y_test_pred_3_5=np.array(predict(w,b,X_test_3_5))

print(y_train_pred_3_5.shape)

train_wrong =(y_train_pred_3_5 != Y_train_3_5)
test_wrong= (y_test_pred_3_5 != Y_test_3_5)


print("trian 틀린 갯수",train_wrong.sum())
print("test 틀린 갯수",test_wrong.sum())

train_mask = train_wrong[0]  # (1, 292) → (292,)
test_mask = test_wrong[0]    # (1, 73) → (73,)

train_wrong_value = X_train_3_5[:, train_mask]
test_wrong_value = X_test_3_5[:, test_mask]

train_wrong_label = Y_train_3_5[:, train_mask]
test_wrong_label = Y_test_3_5[:, test_mask]

images = np.concat([train_wrong_value, test_wrong_value], axis=1)
label = np.concat([train_wrong_label, test_wrong_label], axis=1)

w = result_3_5["w"]
b = result_3_5["b"]
fig ,axes=plt.subplots(1,images.shape[1],
                        figsize=(3*images.shape[1],3),
                        squeeze=False )
for i,ax in enumerate(axes[0]):
    Z=w.T@images[:,i]+b
    A=sigmoid(Z)
    ax.imshow(images[:,i].reshape(8,8),cmap="gray")
    if(label[0,i]==0):
        label_true=3
    elif(label[0,i]==1):
        label_true=5
    ax.set_title(
            f"label={label_true}, p(label=5)={A.item():.3f}"
    )
    ax.axis("off")

plt.tight_layout()
plt.show()

