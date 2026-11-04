# Shallow neural network

## 1. 구현 요약 

## 2. 수식 유도 

**손실 함수**

- $\log \hat y   $

```math
\log\hat y =z-\log(1+e^z)

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
