# Deep Neural Networks

## 1. 구현 요약 


## 2. 검증 결과 

### 2.1 기울기 검증 오차 

코드 

```
    error=gradient_checks_test(X_train,Y_train)
    print("기울기 오차: ", error)
```

결과 
> 기울기 오차: 4.425873650008288e-11

<hr>

### 2.2 환원 검증 

**심층 신경망** 

코드 

```
    layer_dims = [2, 1]

    parameters = {}
    for l in range(1,len(layer_dims)):

        parameters[f"W{l}"] = np.zeros((layer_dims[l],layer_dims[l-1]))
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))


    parameters,history=L_layer_model(X_train, Y_train, layer_dims, 30000, 0.2, X_test, Y_test,parameters)
    print("환원 훈련 정확도: ", history["train_accuracy"][-1])
    print("환원 정확도: ",history["test_accuracy"][-1])
```

결과 
> 환원 훈련 정확도:  0.615 <br>
> 환원 정확도:  0.617

<hr>

**로지스틱 회귀**

코드
```
    def predict_fn(X_grid):
        w, b = initialize(X_train.shape[0])

        params, _, _ = optimize(
            w, b, X_train, Y_train,
            num_iterations=30000,
            learning_rate=0.2
        )

        return logistic_predict(params["w"], params["b"], X_grid)

    predict_train=predict_fn(X_train)
    train_accuracy=np.mean(predict_train == Y_train)
    print("로지스틱 회귀 훈련 정확도: ",train_accuracy)
    
    predict_test=predict_fn(X_test)
    test_accuracy=np.mean(predict_test == Y_test)
    print("로지스틱 회귀 시험 정확도: ",test_accuracy)
```

결과 
> 로지스틱 회귀 훈련 정확도:  0.615 <br>
> 로지스틱 회귀 시험 정확도:  0.617

**일치 여부** 

0으로 초기화 된 심층 신경망의 정확도가 로지스틱 회귀랑 동일하기에 일치한다고 할 수 있다.



<hr>

### 2.3 [2,16,16,1] 시험 정확도

코드 

```
    layer_dims = [2, 16, 16, 1]
    parameters,history=L_layer_model(X_train, Y_train, layer_dims, 30000, 0.2, X_test, Y_test)
    print("훈련 정확도: ", history["train_accuracy"][-1],"시험 정확도: ",history["test_accuracy"][-1])
```

결과 
> 훈련 정확도: 0.9975 시험 정확도:  0.951

<hr>

### 2.4 비용곡선 


![Alt text](../../../images/04-deep-nn/deep_cost_train_02.png)

<hr>


## 3. 실험 A

### 3.1 결과 표 

| # | 설정 | 파라미터 수 | 학습 정확도 | 시험 정확도 | 시험 정확도 90% 도달 | 시험 비용 최소 지점 |
| :-- | :-- | --: | --: | --: | --: | --: |
| A-1 | `[2, 1]` | 3 | 61.5% | 61.7% | - | 607회 |
| A-2 | `[2, 16, 1]` | 65 | 72.5% | 70.1% | - | 8,223회 |
| A-3 | `[2, 64, 1]` | 257 | 62% | 59.6% | - | 3,080회 |
| A-4 | `[2, 256, 1]` | 1,025 | 98.5% | 94.1% | 23400회 | 29,999회 |
| A-5 | `[2, 16, 16, 1]` | 337 | 99.75% | 95.6% | 3,200회 | 5,945회 |
| A-6 | `[2, 32, 16, 8, 1]` | 769 | 100.0% | 95% | 1,200회 | 3,361회 |

<hr>

**A-1, A-5, A-6 결정경계**

![Alt text](../../../images/04-deep-nn/A-1,A-5,A-6%20결정경계2.png)


<hr>

### 3.2 추가 학습 


| # | 학습 정확도 | 시험 정확도 | 시험 정확도 90% 도달 | 시험 비용 최소 지점 |
| :-- | --: | --: | --: | --: |
| A-4 | 98.5% | 94.1% | 23,400회 | 29,999회 |
| A-4 추가 실험 | 99.5% | 95.3% | 23,400회 | 30,745회 |

<hr>

### 3.3 시드 설정 

| # | seed 0 | seed 1 | seed 2 | 
| :-- | --: | --: | --: | 
| A-3 | 59.6% | 65% | 72.6% |
| A-5 | 95.6% | 94.9% | 95.3% |


<hr>

### 생각 해볼 점 


## 4 실험 B

### 4.1 결과 표 

| # | layer_dims | He 초기화 | randn  | 
| :-- | --: | --: | --: | 
| B-1 | [2, 16, 1]   | 70.1% | 61.4% |
| B-2 | [2, 16, 16, 1] | 95.1% | 94.3% |
| B-3 | [2, 32, 16, 8, 1] | 94.6% |	62.7% |

<hr>

### 4.2 비용 곡선 



![Alt text](../../../images/04-deep-nn/deep_cost_train_b_1_2_3.png)




<hr>

### 4.3 생각해볼 점




