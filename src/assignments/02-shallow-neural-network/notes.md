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

## 5. 실험 1

**정리 표**

| 실험 | 학습 정확도 | 시험 정확도 | 최종 학습 비용 | 최종 시험 비용 | 최소 시험 비용 | 최소 비용 반복 | 상승폭 | 판정 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A-1 | 87.75% | 86.40% | 0.297467 | 0.287916 | 0.287916 | 20,000 | 0.000000 | 과소 적합 |
| A-2 | 96.25% | 95.90% | 0.084562 | 0.094869 | 0.091597 | 11,141 | 0.003272 | 정상 |
| A-3 | 97.25% | 96.30% | 0.082535 | 0.086436 | 0.085794 | 7,068 | 0.000643 | 정상 |
| A-4 | 100.00% | 84.90% | 0.022079 | 1.058637 | 0.284483 | 542 | 0.774154 | 정상 |

<hr>

**그래프**

결정 경계

![Alt text](../../../images/03-shallow_nn/실험_A_결정경계.png)

손실 곡선

![Alt text](../../../images/03-shallow_nn/실험_A_손실곡선.png)

<hr>

**과적합 설명**


