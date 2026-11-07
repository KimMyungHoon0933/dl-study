# Shallow neural network

## 1. 구현 요약 

곡선 함수를 구현하는 데이터셋을 기반으로 얇은 신경망과 로지스틱 회귀의 성능차이를 검증하며 
이전 과제에서의 손실함수 값을 다시 검증하였다. 

## 2. 수식 유도 

**손실 함수**

- $\log \hat y   $

```math
\begin{aligned}
\log\hat y

&=\log e^z-\log(1+e^z)\
&=z-\log(1+e^z)
\end{aligned}
```

- $\log 1- \hat y$ 

```math
\log(1-\hat y) -\log(1+e^z)

```

- 최종 식 

```math

\begin{aligned}
-\left[y\log\hat y+(1-y)\log(1-\hat y)\right]
&=-\left[y\{z-\log(1+e^z)\}
       -(1-y)\log(1+e^z)\right]\\
&=\log(1+e^z)-yz
\end{aligned}

```

**역전파**

- $dZ^{[2]}$

```math

dZ^{[2]}=A^{[2]} -Y

```

- $dW^{[2]} ,\quad db^{[2]}$

```math
\begin{aligned}
dW^{[2]}=\frac{1}{m}dZ^{[2]}A^{[1]T}\\
db^{[2]}= \frac{1}{m} \sum _i ^m dZ^{[2]}
\end{aligned}
```

- $dZ^{[1]}$

```math

dZ^{[1]}=\frac{\partial\mathcal L}{\partial Z^{[1]}}
=

\frac{\partial\mathcal L}{\partial Z^{[2]}}
\frac{\partial Z^{[2]}}{\partial A^{[1]}}
\frac{\partial A^{[1]}}{\partial Z^{[1]}}

```

```math



\frac{\partial\mathcal L}{\partial Z^{[2]}}
=dZ^{[2]} ,\quad 

\frac{\partial Z^{[2]}}{\partial A^{[1]}}
= W^{[2]} ,\quad 
\frac{\partial A^{[1]}}{\partial Z^{[1]}}
=1-(A^{[1]})^2

```

```math

dZ^{[1]}
=
\left((W^{[2]})^TdZ^{[2]}\right)
*
1-(A^{[1]})2

```

- $dW^{[1]} ,\quad db^{[1]}$

```math
\begin{aligned}
dW^{[1]}=\frac{1}{m}dZ^{[1]}X^T
\\
db^{[1]}=\frac{1}{m}dZ^{[1]}
\end{aligned}
```


## 3. 구현 검증 결과

### 3.1 기울기 검증 상대 오차

**nh=4**

코드 

```
parameters=initialize_parameters(X_train.shape[0],4,Y_train.shape[0])
error_nh_4=gradient_checks_test(X_train,Y_train,parameters)

print("n_h=4 기울기 오차: ", error_nh_4)
```


결과
> n_h=4 기울기 오차:  4.555480444964085e-11

**nh=50**

코드 

```
parameters=initialize_parameters(X_train.shape[0],50,Y_train.shape[0])
error_nh_50=gradient_checks_test(X_train,Y_train,parameters)

print("n_h=50 기울기 오차: ", error_nh_50)
```

결과
> n_h=50 기울기 오차:  2.985529073064345e-11

<hr>

### 3.2 n_h=4 시험 정확도

![Alt text](../../../images/03-shallow_nn/정확도_곡선.png)

<hr>

### 3.3 결정 경계

![Alt text](../../../images/03-shallow_nn/얇은_신경망_결정경계.png)


## 4. 로지스틱 회귀의 실패 

**시험 정확도**

코드 

```
predict_test=predict_fn(X_test)
test_accuracy=np.mean(predict_test == Y_test)
print("시험 정확도: ",test_accuracy)
```

결과 
> 시험 정확도:  0.862

<hr>

**로지스틱 회귀 결정 경계**

![Alt text](../../../images/03-shallow_nn/결정%20그림.png)


**무엇이 문제인가?**

로지스틱 회귀의 경우 은닉층이 없다. <br>
이는 은닉층에서 출력층 사이의 비선형 함수층이 없어 선형 결합만으로 결정 경계를 형성하게 되어 
직선형태로 만들어지게 된다. <br>
따라서 정확도가 일정 이상 형성되지 못한다.  

## 5. 실험 A

**정리 표**

| 실험 | 학습 정확도 | 시험 정확도 | 최종 학습 비용 | 최종 시험 비용 | 최소 시험 비용 | 최소 비용 반복 | 상승폭 | 판정 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A-1 | 87.75% | 86.40% | 0.297467 | 0.287916 | 0.287916 | 20,000 | 0.000000 | 과소 적합 |
| A-2 | 96.25% | 95.90% | 0.084562 | 0.094869 | 0.091597 | 11,141 | 0.003272 | ? |
| A-3 | 97.25% | 96.30% | 0.082535 | 0.086436 | 0.085794 | 7,068 | 0.000643 | 정상 |
| A-4 | 100.00% | 84.90% | 0.022079 | 1.058637 | 0.284483 | 542 | 0.774154 | 과적합 |

<hr>

**뚜렷하게 기준**

과적합이 나오는 기준은 이 둘의 격차가 급격하게 커지는 것을 기준으로 잡았다. <BR>
정확하게는 A-2의 경우에는 0.003으로 경미한 차이가 할 수 있지만 0.77로 0.1 이상이 낫기에 A-4를 과적합으로 하였다.


<hr>

**그래프**

결정 경계

![Alt text](../../../images/03-shallow_nn/실험_A_결정경계.png)

손실 곡선

![Alt text](../../../images/03-shallow_nn/실험_A_손실곡선.png)

<hr>

**과적합 설명**

A-2과 A-3 비교 

A-2와 A-3는 거의 동일한 정확도를 가지게 된다. 그 이유는 이 둘의 결정 경계를 보면 알 수 있다.<BR>
A-3의 결정 경계가 좀 더 곡률이 높은 곡선을 가지게 되어 정확도가 약간 높지만 경계 함수 자체의 전체적인 형상이 비슷하여 이 둘의 정확도가 비슷하다.  <BR>
하지만 손실곡선과 결정 경계를 보면 차별점이 존재하게 된다. <BR>
A-2의 경우 처음에는 높은 손실을 가지지만 A-3의 경우 값이 처음부터 작으며 빠르게 수렵하게 된다.

A-3와 A-4 비교 

데이터량이 작아지게 되면서 데이터 분포안에 노이즈 까지 학습하게 되면서 <BR>
A-4에서 과적합 현상이 일어나게 되었다.

과적합이란? 

과적합은 모델의 표현력과 학습 데이터의 양·잡음·대표성 사이의 관계에서, 모델이 학습 표본의 노이즈까지 표현하게 되어 새로운 데이터에 대한 일반화 성능이 나빠지는 것을 말한다.

A-2는 과적합인가?

A-2의 경우에는 최대 - 최소의 차이가 0.003으로 차이 값이 커지지 않지만 손실곡선을 보면 계속 올라가고 있기에 이후 학습을 계속 진행하게 된다면 충분한 과적합이 나올 가능성이 있다.

## 6. 실험 B


**결과 표**

| 실험 | 서로 다른 은닉 유닛 수(W1) | 서로 다른 은닉 유닛 수(W2) | 시험 정확도 |
|---|---:|---:|---:|
| B-1 | 1개 | 1개 | 50.0% |
| B-2 | 50개 | 50개 | 96.2% |
| B-3 | 50개 | 50개 | 96.4% |
| B-4 | 1개 | 1개 | 86.3% |

**손실 곡선**

![Alt text](../../../images/03-shallow_nn/실험_B_손실곡선.png)

**


## 7. 실험 C

**결과 표**

| 실험 | 은닉 유닛 수 | 시험 정확도 |
|---|---:|---:|
| C-1 | 4개 | 86.3% |
| C-2 | 50개 | 86.3% |

**결정 경계**

![Alt text](../../../images/03-shallow_nn/실험_C_결정경계.png)

**왜 직선인가?**

## 8. 실험 D

**손실 곡선**

![Alt text](../../../images/03-shallow_nn/실험_D_손실곡선.png)
