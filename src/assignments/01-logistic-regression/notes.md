# logistic regresssion

## 수식 유도 

**순전파**

- 데이터셋의 형태 

$$
{\Large X \in \mathbb{R}^{n \times m}, \quad Y \in \mathbb{R}^{1 \times m}}
$$

- Z

$$
{ \Large Z= W^TX +B; \quad W \in \mathbb{R}^{1\times n}, b \in \mathbb{R}^{1\times m}}
$$


- a

$$
{\Large a =  \sigma(z) ; \quad where \left \sigma(Z) = \frac{1}{1+e^-{z}}}
$$

- J(W,B)

$$
{\Large J(W,B)= -\frac{1}{m}  \sum_{i=1}^{m} \left[y^{(i)}\log a^{(i)} +(1-y^{(i)})\log(1-a^{(i)})\right]}
$$

**역전파**

- $\frac{\partial J(w,b)}{\partial a}$

$$
{\Large \frac{\partial L}{\partial a} 
     = -(y \frac{1}{a} - (1-y)  \frac{1}{1-a})=\frac{a-y}{a(1-a)}}
$$


- 

