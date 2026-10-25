# logistic regresssion

## 1. 구현 요약

NumPy를 사용하여 로지스틱 회귀의 순전파와 역전파를 직접 구현하고 검증하였다.<br>
이를 통해 예측값으로부터 비용과 기울기를 계산하고, 가중치와 편향이 갱신되는 학습 과정 전반을 이해하였다.

## 2. 수식 유도 

**순전파**

- 데이터셋의 형태 

$$
{\Large X \in \mathbb{R}^{n \times m}, \quad Y \in \mathbb{R}^{1 \times m}}
$$

- Z

$$
{ \Large Z= w^TX +B; \quad w \in \mathbb{R}^{n\times 1}, B \in \mathbb{R^{1\times m}}}
$$


- A

$$
{\Large A =  \sigma(Z) ; \quad where \quad \sigma(z) = \frac{1}{1+e^-{z}}}
$$

- J(W,B)

$$
{\Large J(W,B)= -\frac{1}{m}  \sum_{i=1}^{m} \left[y^{(i)}\log a^{(i)} +(1-y^{(i)})\log(1-a^{(i)})\right]}
$$

**역전파**

- $\Large da$

$$
{\Large da=\frac{d L}{d a} 
     = -( y \times \frac{1}{a} - (1-y) \times \frac{1}{1-a})=\frac{a-y}{a(1-a)}}
$$


- $\Large  dz$



```math
{\Large
\begin{aligned}
&\frac{da}{dz} = \frac{e^{-z}}{(1+e^{-z})^2}
= a(1-a)\\
&\rule{0pt}{0.5em}\\
&\begin{aligned}
dz:=\frac{dL}{dz}
&= \frac{dL}{da}\frac{da}{dz}\\
&= \frac{a-y}{a(1-a)}\,a(1-a)\\
&= a-y
\end{aligned}
\end{aligned}
}
```



**$\Large dw ,dw$**

- dZ

$$
{\Large dZ= [dz^{(1)},\ldots,dz^{(m)}]=A-Y}
$$

- dw 식

$$
{\Large
\begin{aligned}
dw =\frac{1}{m}XdZ^T \\
&\rule{0pt}{0.5em}\\
{\Large dw \in \mathbb{R}^{n \times 1}}
\end{aligned}
}
$$



- db 식


$$
{\Large
\begin{aligned}
db =\frac{1}{m}\sum _i ^m dZ \\
&\rule{0pt}{0.5em}\\
{\Large db \in \mathbb{R}}
\end{aligned}
}
$$


## 3. 검증 결과

### 3.1 sigmoid 오버플로우 경고

**원인**

sigmoid에서 오버플로우 경고는 $e^-z$ 에서 z값이 음수 상황에서 값이 클 경우에 발생하게 된다. 

**해결방법**

이를 해결하고자 음수 일때는 $e^z$ 형태로 계산하여 큰값이 반환이 되지 않게 계산을 진행한다. <br>
식은 다음과 같다. 


$$
\sigma(z)=
\begin{cases}
\frac{1}{1+e^{-z}}, & z \ge 0 \\
\frac{e^z}{1+e^z}, & z < 0
\end{cases}
$$



**결과**

코드 

```
def sigmoid(z):
    #예상 입력: z=(1,m)
    z=np.array(z)
    A=np.zeros_like(z, dtype=float)
    positive = (z >= 0)
    
    A[positive]=1 / (1 + np.exp(-z[positive]))

    exp_z = np.exp(z[~positive])
    A[~positive] = exp_z / (1 + exp_z)

    return A #출력 A.shape=(1,m)
overflow_debug=sigmoid(np.array([-1000, 0, 1000]))
print("시그모이드 오버플로우 검증 "overflow_debug)
```

결과 
>> 시그모이드 오버플로우 검증:  [0.  0.5 1. ]

### 3.2 비용함수 안정성 문제

**log(a)**


**해결방법**

**결과**

### 3.3 기울기 검증 상대 오차 


### 3.4 기울기 검증기 디버깅


### 3.5 비용함수 곡선

![Alt text](../../../images/02-neural/cost_function.png)

### 3.6 train / test 정확도 

**코드**

```
result = model(X_train, Y_train, X_test, Y_test)

print(f"훈련 정확도: {result['train_accuracy']:.2%}")
print(f"테스트 정확도: {result['test_accuracy']:.2%}")
```

**출력결과**

> 훈련 정확도: 100.00%
> 테스트 정확도: 100.00%
