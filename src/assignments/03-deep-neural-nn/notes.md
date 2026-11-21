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

**seed에 따른 값의 변화**

이는 초기화 시드에 따라서 가중치가 시작하는 위치 즉 도함수상에서의 시작 위치가 달라지게 되면서 <BR>
도달하게 되는 Minima이 달라지게 된다. 

따라서 신경망의 경우 여러 해가 존재하며 이 해의 따라 성능이 달라지게 된다. 

<hr>

### 생각 해볼 점 

**A-2와 A-4 비교**

A-2와 A-4는 깊이가 같지만 파라미터 수에 차이가 있다.  

이 파리미터 차이로 인해서 A-4가 더 복잡한 결정 경계를 표현할 수 있어, <BR>
나선 데이터에서 A-4가 높은 분리 성능을 보여준게 된다.

**A-3와 A-5 비교**

앞선 방식과 달리 A-3와 A-5를 비교 하였을 때 두 모델의 파라미터 갯수는 비슷하다.  <BR>

하지만 신경망은 깊은 층에서 이전 층의 정보를 기반으로 더욱 복잡한 함수를 표현할 수 있다.
이로 인해 A-5 또한 적은 파리미터에서도 복잡한 선형경계를 만들 수 있다


**A-4와 A-5 비교**

A-4와 A-5는 성능 측면에서 비슷한 성능을 가지만 A-5가 시험 성능에서 약간의 우위를 가지게 된다. <BR>
A-5는 A-4보다 적은 학습 반복으로 테스트 정확도 90%에 도달하였다. <BR>

이는 깊어 질수록 데이터에 따른 규칙에 대하여 효율적으로 표현 가능하게 하여, <br> 
일반화 성능이 좋아지게 되어 좀 더 빠르게 도달한 것으로 보인다<BR>


**깊게 하면 무엇이 좋아지는가?**

앞선 내용을 종합하면, 신경망의 깊이를 늘리는 것은 파라미터 표현 효율과 일반화 성능 측면에서 이점을 줄 수 있다. <BR>

이는 더 깊은 모델은 비슷한 수의 파라미터로 더 복잡한 결정 경계를 표현할 수 있으며, <BR> 
일반화 성능이 좋아지는 것을 보여준다.


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

**두 초기화의 차이**

층이 깊어질수록 randn 초기화 쪽이 점점 업데이트가 안되는 것을 볼 수 있다.

**0.5값을 가지는 모델과 B-3의 학습 값 비교**

코드
```
    cost_05 = compute_cost(np.zeros_like(Y, dtype=float), Y)

    print("항상 0.5를 출력하는 망의 비용:", cost_05)
    print("B-3 randn 최종 학습 비용:", historys_randn[2]["cost_list"][-1])
    return test_accuracy_He,test_accuracy_randn
```

결과 
>  항상 0.5를 출력하는 망의 비용: 0.6931471805599452 <BR>
>  B-3 randn 최종 학습 비용: 0.6931471741078067


**같아 지는 이유**

이는 업데이트가 되지 않아 항상 그 출력값을 가지게 되어 거의 동일한 출력을 가지게 된다.

**가중치를 작게 시작하면 층을 지날 때마다 신호가 어떻게 되는가**

층이 지날수록 가중치값의 의해서 점점 값이 작아지게 된다. 이에 따라 역전파로 업데이트 되는 기울기 또한 작아지게 된다. 


