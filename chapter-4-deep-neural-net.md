---
description: Multi-Class Classification
---

# Chapter 4 / Deep Neural Net

Fasion MNIST 데이터셋, Deep Neural Net을 이용해 패션 아이템(class=10)을 분류하는 모델을 만듭니다.

## Dataset

Fasion MNIST 데이터셋 사용

28x28 픽셀의 70000개의 흑백 이미지로 패션 아이템을 10가지 카테고리로 나눈 데이터셋

MNIST가 숫자 데이터를 28x28, 10개로 나눈 것처럼 비슷한 형태, torchvision에서 데이터셋을 로드

### torchvision, utils

`transform = transforms.Compose([transform리스트])` 로 변환기를 만들고, `torchvision.datasets`에서 데이터셋을 로드할 때 transform 키워드에 transform을 할당해주면 데이터셋을 transform 리스트 순서대로 변환

```python
trainset = datasets.FashionMNIST(root='./.data/', train=True, download=True, transform=transform)
testset = datasets.FashionMNIST(root='./.data/',train=False,download=True,transform=transform)
```

사용된 구문들

| 구문                     | 설명                                      |
| ---------------------- | --------------------------------------- |
| transform.Compose(\[]) | transform 오브젝트를 만들어 일괄적으로 변환하는 변환기 만들기  |
| datasets.FasionMNIST() | fasion MNIST 불러오기                       |
| data.DataLoader()      | dataset 객체를 다양한 옵션으로 불러와 반복 가능한 객체로 만들기 |

## Model

다음 코드로 CUDA 확인

```python
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
```

모델의 구조는 다음과 같이 제작

| Layer  | dimension        |
| ------ | ---------------- |
| view   | 28\*28\*1 -> 784 |
| Linear | 784 -> 256       |
| ReLU   |                  |
| Linear | 256 -> 128       |
| ReLU   |                  |
| Linear | 128 -.> 10       |

`model.to(DEVICE)` 를 통해 CUDA를 사용할 수 있음

`torch.nn.Module` 로부터 서브클래싱해 만든 모델이다. nn.Module을 사용하면 직관적으로 커스텀 모델을 만들 수 있다.

```python
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(784,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,10)

    def forward(self,x):
        x = x.view(-1,784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

## Optimization

| 항목            |  값            |
| ------------- | ------------- |
| Epochs        | 30            |
| Batch Size    | 64            |
| Loss Function | Cross Entropy |
| Optimizer     | SGD           |
| Learning Rate | 0.01          |

다음 코드는 train, evaluate 부분아다.

```python
def train(model, train_loader, optimizer):
    model.train() #모드 변경
    for batch_idx, (data, target) in enumerate(train_loader): #모든 미니배치 반복
        data, target = data.to(DEVICE), target.to(DEVICE) #데이터 이동
        optimizer.zero_grad() #grad 초기화
        output = model(data) #forward propagation
        loss = F.cross_entropy(output,target) # loss 계산
        loss.backward() #gradient 계산
        optimizer.step() #optimize

def evaluate(model, test_loader):
    model.eval() #모드 변경
    test_loss = 0
    correct = 0
    with torch.no_grad(): #gradient 계산 안하는 scope
        for data, target in test_loader:
            data,target = data.to(DEVICE), target.to(DEVICE)
            output = model(data) 
            test_loss += F.cross_entropy(output,target,reduction='sum').item() #reduction='sum'으로 배치 원소들의 합 구하기
            pred = output.max(1,keepdim=True)[1] #index 구하기(argmax)
            correct += pred.eq(target.view_as(pred)).sum().item() # view_as로 모양 일치, 모든 배치 원소에 대해 일치하면 1, 모두 합하기
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy
    
for epoch in range(1,EPOCHS+1):
    train(model,train_loader,optimizer) #train
    test_loss, test_accuracy = evaluate(model,test_loader) #test
    print(f'{epoch} Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}')
```

학습 과정은 다음과 같다.

1. data와 target을 GPU(CPU)로 이동
2. optimizer의 gradient를 초기화
3. data에 대한 model의 출력 연산
4. output에 대한 target과의 loss 계산
5. loss를 역미분해 gradient 할당
6. optimizer에 할당된 parameter들의 계산된 gradient를 적용

평가 방법은 다음과 같다.

1. data와 target을 GPU(CPU)로 이동
2. data에 대한 model의 출력 연산
3. 출력과 target의 loss 계산, 모든 batch에 대해 합함
4. 출력의 argmax를 구해 얼마나 많이 정답을 맞혔는지 연산



## Data Augmentation

이미지를 무작위로 뒤집어 데이터셋의 크기를 늘려 학습을 더 잘 할 수 있게 만든다. 본 챕터에서 RandomHorizontalFlip을 transform에 추가해 데이터를 늘린다.

## Dropout

과적합은 적은 데이터에 대해 예측이 학습 오차를 줄이는 데 과하게 학습하고 실제 일반적 데이터에 대해선 일반화하지 못하는 경우를 말한다. train loss는 계속 줄어들지만, validation loss는 증가하는 시점이 있는데, 이 시점에서 학습을 종료해야 적절한 예측을 얻을 수 있다.

dropout은 신경망에서 다음 layer로 이동할 때 일정 확률로 node가 없는 것처럼 이동한다. 계산한 결과에서 일부 node의 결과를 일정 확률로 제거해 과적합을 방지한다.&#x20;

간단히 dropout 함수를 거침으로써 사용할 수 있다.

