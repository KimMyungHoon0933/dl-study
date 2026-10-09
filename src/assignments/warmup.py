import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 과제 1
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a,b)
toc = time.time()

print(c)
print("vectorized version " + str(1000*(toc-tic))+" ms")

tic = time.time()
for i in range(1000000):
    c += a[i]*b[i]
toc = time.time()

print(c)
print("for loop " + str(1000*(toc-tic))+" ms")


# 과제 2
a=np.random.rand(3, 4)
print(a/a.sum(axis=0,keepdims=False))
print(a/a.sum(axis=0,keepdims=True))

#과제 3
a = np.random.randn(5)
b = np.random.randn(5, 1)
print("a.shape: " ,a.shape , "b.shape: ", b.shape)
print("a.T.shape: " ,a.T.shape , "b.T.shape: ", b.T.shape)
print("np.dot(a, a.T): ",np.dot(a, a.T),"np.dot(b, b.T): ", np.dot(b, b.T))


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

plt.plot(x,y1)
plt.scatter(y1_0, 0, color='blue', s=60)
plt.plot(x,y2)
plt.scatter(y2_0, 0, color='orange', s=60)
plt.plot(x,y3)
plt.scatter(y3_0, 0, color='green', s=60)
plt.show()

c1=3
c2=1
c3=-4
y4 = c1*relu(w1*x+b1) + c2*relu(w2*x+b2) + c3*relu(w3*x+b3)
plt.plot(x,y4)
c1=4
c2=5
c3=6
y5 = c1*relu(w1*x+b1) + c2*relu(w2*x+b2) + c3*relu(w3*x+b3)
plt.plot(x,y5)
c1=2
c2=-3
c3=-5
y6 = c1*relu(w1*x+b1) + c2*relu(w2*x+b2) + c3*relu(w3*x+b3)
plt.plot(x,y6)
plt.show()


result_10=relu_for_result(x,10)
plt.plot(x,result_10)

result_50=relu_for_result(x,50)
plt.plot(x,result_50)

result_100=relu_for_result(x,100)
plt.plot(x,result_100)

plt.show()

result_10=linear_for_result(x,10)
plt.plot(x,result_10)

result_50=linear_for_result(x,50)
plt.plot(x,result_50)

result_100=linear_for_result(x,100)
plt.plot(x,result_100)

plt.show()



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

plt.plot(x,y_true)
plt.plot(x,y_pred)
plt.show()

H= H_calculater(x,w,b,20)
c=np.linalg.lstsq(H, y_true, rcond=None)[0]
y_pred=H@c

plt.plot(x,y_true)
plt.plot(x,y_pred)
plt.show()

H= H_calculater(x,w,b,100)
c=np.linalg.lstsq(H, y_true, rcond=None)[0]
y_pred=H@c

plt.plot(x,y_true)
plt.plot(x,y_pred)
plt.show()



n_array = np.arange(5, 201)
mse_array=[]

for i in n_array:
    H = H_calculater(x,w,b,i)
    c = np.linalg.lstsq(H, y_true, rcond=None)[0]
    y_pred=H@c
    mse_array.append( mse(y_pred,y_true))
print( mse_array)
plt.plot(n_array,mse_array)
plt.yscale("log")
plt.show()
    

