# logistic regresssion

## 수식 유도 

**순전파**

- 데이터셋의 형태 

$$
{\Large X \in \mathbb{R}^{n \times m}, \quad Y \in \mathbb{R}^{1 \times m}}
$$

- Z

$$
{ \Large Z= w^TX +B; \quad w \in \mathbb{R}^{n\times 1}, b \in \mathbb{R}}
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

- $da$

$$
{\Large da=\frac{d L}{d a} 
     = -( y \times \frac{1}{a} - (1-y) \times \frac{1}{1-a})=\frac{a-y}{a(1-a)}}
$$


- ${dz}$

$$
{\Large dz = \frac{dL}{da}\frac{da}{dz}}
$$

$$
{\Large\frac{da}{dz}= \frac{e^{-z}}{(1+e^{-z})^2}=a(1-a)}
$$

$$
\Large dz= \frac{a-y}{a(1-a)} \times a(1-a) = a-y
$$


**$dw ,dw의 식과 shape$**

- dZ

$$
{\Large dZ= \sum _i^m dz^{(i)}=A-Y}
$$

- dw 식

$$
{\Large dw += \frac{1}{m}XdZ^T}
$$

- dw.shape

$$
 \underbrace{X}_{n\times m} 
\underbrace{Z^T}_{m\times 1}
=
\underbrace{dXZ^T}_{n\times 1}
$$

$$
{\Large dw \in \mathbb{R}^{1 \times n}}
$$

- db 식

$$
db += dZ
$$

- db shape

$$
{\Large db \in \mathbb{R}^{1 \times m}}
$$