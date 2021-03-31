---
description: Multi-Class Classification
---

# Chapter 4 / Deep Neural Net

Fasion MNIST 데이터셋, Deep Neural Net을 이용해 패션 아이템\(class=10\)을 분류하는 모델을 만듭니다.

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

| 구문 | 설명 |
| :--- | :--- |
| transform.Compose\(\[\]\) | transform 오브젝트를 만들어 일괄적으로 변환하는 변환기 만들기 |
| datasets.FasionMNIST\(\) | fasion MNIST 불러오기 |
| data.DataLoader\(\) | dataset 객체를 다양한 옵션으로 불러와 반복 가능한 객체로 만들기 |

## Model

다음 코드로 CUDA 확인

```python
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
```

모델의 구조는 다음과 같이 제작

| Layer | dimension |
| :--- | :--- |
| view | 28\*28\*1 -&gt; 784 |
| Linear | 784 -&gt; 256 |
| ReLU |  |
| Linear | 256 -&gt; 128 |
| ReLU |  |
| Linear | 128 -.&gt; 10 |

`model.to(DEVICE)` 를 통해 CUDA를 사용할 수 있음



## Optimization

batch size = 64, epochs = 30으로 설정, Optimizer는 SGD\(Stochastic Gradient Descent\), Learning rate는 0.01로 설정, loss function은 Cross Entropy를 사용

다음 코드는 train, evaluate 

```python
def train(model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output,target)
        loss.backward()
        optimizer.step()

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data,target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += F.cross_entropy(output,target,reduction='sum').item()
            pred = output.max(1,keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy
    
for epoch in range(1,EPOCHS+1):
    train(model,train_loader,optimizer)
    test_loss, test_accuracy = evaluate(model,test_loader)
    print(f'{epoch} Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}')
```

