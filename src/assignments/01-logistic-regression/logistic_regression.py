# 0. 데이터 셋 준비
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
    return np.exp(-np.logaddexp(0, -z))

overflow_debug=sigmoid(np.array([-1000, 0, 1000]))
print(overflow_debug)

#initialize
def initialize(dim):
    w=np.zeros((dim,1))
    b=np.zeros(1)
    return w,b

#w,b= initialize(64)
#print("w shape:",w.shape,"b shape: ",b.shape)

#propagate
def cost_function(a,Y,m):
    L=-(a*np.log(a)+(1-Y)* np.log(1-a))
    J=np.sum(L)/m
    return J

def propagate(w,b,X,Y,m):
    Z=w.T@ X + b
    A=sigmoid(Z)
    cost=cost_function(A,Y,m)
    dZ= A-Y 
    dw= (X @ dZ.T)/m
    db= np.mean(dZ)
    return dw,db,cost

w,b= initialize(64)
w,b,cost=propagate(w,b,X_train,Y_train,288)
print(w.shape,b.shape,cost.shape)

    