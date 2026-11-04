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

dZ^{[1]}=\frac{\partial\mathcal L}{\partial z^{[1]}}
=

\frac{\partial\mathcal L}{\partial z^{[2]}}
\frac{\partial z^{[2]}}{\partial a^{[1]}}
\frac{\partial a^{[1]}}{\partial z^{[1]}}

```

```math



\frac{\partial\mathcal L}{\partial z^{[2]}}
=dz^{[2]} ,\quad 

\frac{\partial z^{[2]}}{\partial a^{[1]}}
= g^{[1]\prime}(z^{[1]}) ,\quad 
\frac{\partial a^{[1]}}{\partial z^{[1]}}
=W^{[2]}

```

```math

dz^{[1]}
=
\left((W^{[2]})^Tdz^{[2]}\right)
*
g^{[1]\prime}(z^{[1]})

```

- $dW^{[1]} ,\quad db^{[1]}$

```math
\begin{aligned}
dW^{[1]}=dz^{[1]}x^T
\\
db^{[1]}=dz^{[1]}
\end{aligned}
```

