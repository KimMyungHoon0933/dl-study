# Shallow neural network

## 1. 구현 요약 

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

