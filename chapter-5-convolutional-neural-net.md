---
description: CNN
---

# Chapter 5 / Convolutional Neural Net

CNN은 컨볼루션 연산을 이미지에 적용해 영상 처리에 최적화한 뉴럴 네트워크이다. 이미지의 픽셀 하나하나가 아닌 각 부분별로 필터를 적용해 뉴럴 네트워크를 구성한다. 필터를 이미지의 모든 부분에 곱하고 더해 값을 출력한다.&#x20;

CNN은 보통 컨볼루션 계층과 풀링 계층으로 이루어진다. 컨볼루션 계층은 컨볼루션 연산을 적용하고, 풀링 계층은 컨볼루션한 이미지에서 중요 특성만 골라 이미지의 차원을 축소시킨다. 컨볼루션 계층엔 다양한 옵션이 있으며 이것에 따라 출력 채널과 이미지 크기가 달라진다.

## CNN with FashionMNIST

### Model

| Layer            | Info                           |
| ---------------- | ------------------------------ |
| Conv2D           | 1->10, kernel\_size=5          |
| MaxPool2D+ReLU   | kernel\_size=2                 |
| Conv2D+Dropout2D | 10->20, kernel\_size=5 , p=0.5 |
| MaxPool2D+ReLU   | kernel\_size=2                 |
| Linear+ReLU      | 320->50                        |
| Dropout          | p=0.5                          |
| Linear           | 50->10                         |

torch.nn.Module에서 서브클래싱한 CNN 모델 코드

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self ).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.drop = nn.Dropout()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10)

    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = x.view(-1,320)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x
```

### Hyperparameters & etc.

| 항목            | 값                         |
| ------------- | ------------------------- |
| Epochs        | 40                        |
| Batch Size    | 64                        |
| Loss Function | Catecorical Cross Entropy |
| Optimizer     | SGD                       |
| Learning Rate | 0.01                      |
| Momentum      | 0.5                       |

loss function은 Cross Entropy를 사용한다. Optimizer는 SGD에 momentum을 적용해 사용한다. learning rate는 0.01, momentum은 0.5이다.

#### **중요: 해당 챕터의 결과는 MNIST를 사용하지만, 설명은 FashionMNIST를 사용하는 것으로 나와있다. 필자의 착오로 인해 잘못 쓴 것 같다. FashionMNIST를 사용하면 책 저자의 코드로도 99%가 나오지 않는다.**&#x20;



