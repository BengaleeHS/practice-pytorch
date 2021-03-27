# Chapter 3 / ANN

## Tensors

| Type | e.g. |
| :--- | :--- |
| Scalar | 1 |
| Vector \(Rank 1 tensor\) | \[1 2 3\] |
| Matrix \(Rank 2 tensor\) | \[\[1, 2 \], \[3, 4\]\] |
| Tensor \(Rank n tensor\) | \[\[\[1,2,3\]\]\] |

Tensor를 만들 때는 

```python
x = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
```

tensor의 members

| Attribute | Desc. |
| :--- | :--- |
| size\(\) : method | 텐서 형태 리턴 |
| shape : property | 텐서 형태 프로퍼티\(ndarray.shape?\) |
| ndimension\(\) : method | 텐서 랭크 리턴 |
| view\(\*list\) : method | np.reshape\(\) |

tensor에 가할 수 있는 조작들

| method | Desc. |
| :--- | :--- |
| torch.unsqueeze\(tensor, n\) | n 랭크 위치에서 랭크 1 늘려 리턴 |
| torch.squeeze\(tensor\) | 랭크 1 줄이기 |

연산들

| method | Desc. |
| :--- | :--- |
| torch.mm\(a,b\) | = matmul\(\) |
| torch.cat\(tensors, dim\) | dim 기준으로 tensor를 concat |
| torch.dist\(a,b\) | a, b 텐서 사이의 거리 |

## Autograd

Gradient Descent 자동 구현

Gradient 적용 방법  -  텐서 선언 시 다음과 같이 requires\_grad 를 True로 설정하면 tensor.grad에 gradient를 저장 

```python
w = torch.tensor(1.0, requires_grad=True)
```

예시\(backward\(\)\)

```python
a = (w*3)**2
a.backward() #미분 역계산
print(w.grad) #미분값 출력
```

간단한 GD

```python
lr = 0.8 # learning rate
random_tensor = torch.randn(10000, dtype = torch.float)

for i in range(20000): # 반복
    random_tensor.requires_grad_(True) #gradient 저장 ON
    hypothesis = weird_function(random_tensor)
    loss = dist_loss(hypothesis,broken_img) # loss 계산
    loss.backward() # backpropagation

    with torch.no_grad(): # gradient 계산 없이 진행하는 scope
        random_tensor = random_tensor - lr*random_tensor.grad
```

## ANN

$$
\mathbf{y}=f(\mathbf{W} \cdot \mathbf{x} + \mathbf{b})
$$

가중치 행렬에 이전 층의 벡터를 곱하고 bias를 더하고 비선형화를 위해 활성화함수 적용

`torch.nn.Linear( in, out)`으로 wx+b, `torch.nn.ReLU()`로 ReLU, `torch.nn.Sigmoid()`로 시그모이드 활성화 함수 적용

`torch.nn.BCELoss()`로 Binary Cross Entropy loss를 계산하는 객체를 생성, `torch.optim.SGD(parameters,lr)`로 해당 파라미터를 대상으로 하는 optimizer를 생성.

`model.eval()`과 `model.train()`은 각각 실행 모드와 학습 모드를 스위칭할 수 있다.

`optimizer.zero_grad()`를 이용해 gradient를 0으로 재설정해 새로운 gradient를 계산할 수 있도록 함.

train 모드로 변경한 후, 

모델에 데이터를 넣어 output계산, loss를 계산, loss를 backpropagation, optimizer으로 parameter에 저장된 gradient를 이용해 최적화 - 를 반혹한다. 

