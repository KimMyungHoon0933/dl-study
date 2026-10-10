import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 과제 1
print("\n\n\n\n\n")
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic_dot = time.time()
c_dot = np.dot(a,b)
toc_dot = time.time()

print("np.dot을 통한 결과값: ",c_dot)
print("넘파이 dot 걸린 시간: " + str(1000*(toc_dot-tic_dot))+" ms")

c_for=0
tic_for = time.time()
for i in range(1000000):
    c_for += a[i]*b[i]
toc_for = time.time()

print("for을 통한 결과값: ",c_for)
print("for loop 걸린 시간: " + str(1000*(toc_for-tic_for))+" ms")
print(c_dot==c_for)
print(np.allclose(c_dot,c_for))

print("\n\n\n\n\n")



# 과제 2
a=np.random.rand(3, 4)
print("열 정규화 keepdims False 결과값: \n",a/a.sum(axis=0,keepdims=False))
print("\n\n\n\n\n")
print("열 정규화 keepdims True 결과값: \n",a/a.sum(axis=0,keepdims=True))
print("\n\n\n\n\n")
print("각 열의 합: ",(a/a.sum(axis=0,keepdims=True)).sum(axis=0,keepdims=True))
print("\n\n\n\n\n")
print("각 열 검사:", np.isclose((a/a.sum(axis=0,keepdims=True)).sum(axis=0,keepdims=True), 1))
print("\n\n\n\n\n")



#print("행 정규화 keepdims False 결과값: \n",a/a.sum(axis=1,keepdims=False))
print("행 정규화 keepdims True 결과값: \n",a/a.sum(axis=1,keepdims=True))
print("\n\n\n\n\n")
print("각 행의 합: ",(a/a.sum(axis=1,keepdims=True)).sum(axis=1,keepdims=True))

print("\n\n\n\n\n")
print("각 열 검사:", np.isclose((a/a.sum(axis=1,keepdims=True)).sum(axis=1,keepdims=True), 1))
print("\n\n\n\n\n")

#과제 3
a = np.random.randn(5)
b = np.random.randn(5, 1)
print("a.shape: " ,a.shape , "   b.shape: ", b.shape)
print("\n\n\n\n\n")
print("a.T.shape: " ,a.T.shape , "   b.T.shape: ", b.T.shape)
print("\n\n\n\n\n")
print("np.dot(a, a.T): ",np.dot(a, a.T),"\n\n\n\nnp.dot(b, b.T):\n ",np.dot(b, b.T))
print("\n\n\n\n\n")


#과제 4
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    x=np.maximum(0,x)
    return x 

def relu_for_result(x,relu_number):
    w = np.random.randn(relu_number)
    c = np.random.randn(relu_number)
    r = np.random.uniform(-5, 5, relu_number)

    b = -w * r # 꺾이는 위치 r을 먼저 [-5,5] 사이 값으로 정하고, 이미 알고 있는 w를 이용해서 그 위치에서 꺾이게 하는 b를 구하는 식임.
    
    result=np.zeros(200)
    

    for i in range(relu_number):
        activate=relu(w[i]*x+b[i])
        result+=c[i]*activate
    return result

def linear_for_result(x,relu_number):
    w = np.random.randn(relu_number)
    c = np.random.randn(relu_number)
    r = np.random.uniform(-5, 5, relu_number)

    b = -w * r # 꺾이는 위치 r을 먼저 [-5,5] 사이 값으로 정하고, 이미 알고 있는 w를 이용해서 그 위치에서 꺾이게 하는 b를 구하는 식임.
    
    result=np.zeros(200)
    

    for i in range(relu_number):
        activate=w[i]*x+b[i]
        result+=c[i]*activate
    return result


x = np.linspace(-5, 5, 200)
y=relu(x)
plt.plot(x, y)
plt.show()

w1=2 
b1=3
y1=relu(w1*x + b1)
y1_0=-b1/w1

w2=1 
b2=-3
y2=relu(w2*x + b2)
y2_0=-b2/w2

w3=-2 
b3=+1
y3=relu(w3*x + b3)
y3_0=-b3/w3

plt.plot(x,y1,label="ReLU 1", color="blue")
plt.scatter(y1_0, 0, color='blue', s=60)
plt.plot(x,y2,label="ReLU 2", color="orange")
plt.scatter(y2_0, 0, color='orange', s=60)
plt.plot(x,y3,label="ReLU 3", color="green")
plt.scatter(y3_0, 0, color='green', s=60)
plt.legend(loc="upper right")
plt.show()

c1=3
c2=1
c3=-4
y4 = c1*relu(w1*x+b1) + c2*relu(w2*x+b2) + c3*relu(w3*x+b3)
plt.plot(x,y4,label="ReLU sum1", color="blue")
c1=4
c2=5
c3=6
y5 = c1*relu(w1*x+b1) + c2*relu(w2*x+b2) + c3*relu(w3*x+b3)
plt.plot(x,y5,label="ReLU sum2", color="orange")
c1=2
c2=-3
c3=-5
y6 = c1*relu(w1*x+b1) + c2*relu(w2*x+b2) + c3*relu(w3*x+b3)
plt.plot(x,y6,label="ReLU sum3", color="green")
plt.legend(loc="upper right")
plt.show()


result_10=relu_for_result(x,10)
plt.plot(x,result_10,label="ReLU sum10", color="blue")

result_50=relu_for_result(x,50)
plt.plot(x,result_50,label="ReLU sum50", color="orange")

result_100=relu_for_result(x,100)
plt.plot(x,result_100,label="ReLU sum100", color="green")
plt.legend(loc="upper right")
plt.show()

result_10=linear_for_result(x,10)
plt.plot(x,result_10,label="linear sum10", color="blue")

result_50=linear_for_result(x,50)
plt.plot(x,result_50,label="linear sum50", color="orange")

result_100=linear_for_result(x,100)
plt.plot(x,result_100,label="linear sum100", color="green")
plt.legend(loc="upper right")
plt.show()

print("\n\n\n\n\n")


#과제 5
def H_calculater(x,w,b,n):
    H= relu(x[:,np.newaxis] * w[np.newaxis,:n] + b[np.newaxis,:n])
    return H
def c_create(H,relu_number):

    c=np.linalg.lstsq(H[:relu_number], y_true, rcond=None)[0]
    return c

def mse(a,b):
    result=np.mean((a - b) ** 2)
    return result




x = np.linspace(-5, 5, 200)
y_true = np.sin(x)

print(x.shape,y_true.shape)

w = np.random.randn(200)
c = np.random.randn(200)
r = np.random.uniform(-5, 5, 200)
b = -w * r 

H= H_calculater(x,w,b,10)
c=np.linalg.lstsq(H, y_true, rcond=None)[0] #H*c=y가 같아지는 c를 찾는 것이기에 차원이 안맞아도 y_true의 차원이 되게 하는 차원 c를 구하는 것임
y_pred=H@c

plt.plot(x,y_true,label="y_true_n=10",color="blue")
plt.plot(x,y_pred,label="y_pred_n=10",color="green")
plt.legend(loc="upper right")
plt.show()

H= H_calculater(x,w,b,20)
c=np.linalg.lstsq(H, y_true, rcond=None)[0]
y_pred=H@c

plt.plot(x,y_true,label="y_true_n=20",color="blue")
plt.plot(x,y_pred,label="y_pred_n=20",color="green")
plt.legend(loc="upper right")
plt.show()

H= H_calculater(x,w,b,100)
c=np.linalg.lstsq(H, y_true, rcond=None)[0]
y_pred=H@c

plt.plot(x,y_true,label="y_true_n=100",color="blue")
plt.plot(x,y_pred,label="y_pred_n=100",color="green")
plt.legend(loc="upper right")
plt.show()



n_array = np.arange(5, 201)
mse_array=[]

for i in n_array:
    H = H_calculater(x,w,b,i)
    c = np.linalg.lstsq(H, y_true, rcond=None)[0]
    y_pred=H@c
    mse_array.append( mse(y_pred,y_true))

plt.plot(n_array,mse_array)
plt.yscale("log")
plt.show()
    

