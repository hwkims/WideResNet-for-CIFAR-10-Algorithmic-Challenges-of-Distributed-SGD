아래는 **WideResNet** 모델을 포함한 GitHub 리드미 파일 예시입니다. 제목에 `WideResNet`을 추가하여 코드와 관련된 내용을 강조한 버전입니다.
![image](https://github.com/user-attachments/assets/6083e29c-78a7-44c7-b1ac-8102de9b998d)
![image](https://github.com/user-attachments/assets/d0f306db-ab40-489f-8a6b-94e6df676272)

---

# WideResNet for CIFAR-10: Algorithmic Challenges of Distributed SGD

이 프로젝트는 **WideResNet** 모델을 사용하여 **분산 Stochastic Gradient Descent (SGD)** 알고리즘을 구현하는 NVIDIA DLI 과정의 일환입니다. 이 코드는 CIFAR-10 데이터셋을 이용해 **WideResNet**을 훈련시키고, 여러 GPU를 활용하여 분산 학습을 수행하는 예시를 제공합니다. 또한 **조기 종료 (Early Stopping)** 및 **목표 정확도 달성 시 자동 종료** 기능을 포함하고 있습니다.

## 주요 기능

- **WideResNet 모델 구현**: CIFAR-10 이미지 분류를 위한 Residual 네트워크 아키텍처
- **분산 학습 (Data Parallelism)**: 여러 GPU에서 데이터 병렬 처리를 통해 학습 속도 향상
- **조기 종료 (Early Stopping)**: 목표 정확도에 도달하면 훈련을 자동으로 종료
- **CIFAR-10 데이터셋**: 훈련 및 테스트용 데이터셋 제공
- **학습 및 평가 함수**: 훈련 및 테스트를 위한 함수 포함

## 실행 방법

이 코드를 실행하려면 `assessment.py` 파일을 통해 모델을 훈련하고 평가할 수 있습니다. 아래와 같은 명령어로 실행합니다:

```bash
python3 assessment.py --node-id 0 --num-gpus 4 --num-nodes 1 --batch-size 128 --target-accuracy 0.75 --patience 2
```

### 인자 설명:

- `--node-id`: 현재 노드의 ID (기본값: `0`)
- `--num-gpus`: 노드에서 사용할 GPU의 개수 (기본값: `4`)
- `--num-nodes`: 사용할 노드의 수 (기본값: `1`)
- `--batch-size`: 훈련 시 배치 사이즈 (기본값: `128`)
- `--target-accuracy`: 목표 정확도 (기본값: `0.75`)
- `--patience`: 목표 정확도에 도달한 이후 기다릴 에폭 수 (기본값: `2`)

### 예시:

```bash
python3 assessment.py --node-id 0 --num-gpus 4 --num-nodes 1 --batch-size 128 --target-accuracy 0.75 --patience 2
```

## 코드 설명

### 1. `WideResNet` 모델

`WideResNet` 모델은 이미지 분류를 위한 **Residual Network** 아키텍처를 기반으로 하며, 여러 개의 **Residual Block**을 포함하고 있습니다. 각 블록은 **Convolution**, **Batch Normalization**, **ReLU 활성화 함수**를 사용하여 구성됩니다.

```python
class cbrblock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(cbrblock, self).__init__()
        self.cbr = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=(1,1),
                                           padding='same', bias=False), 
                                 nn.BatchNorm2d(output_channels), 
                                 nn.ReLU())
    
    def forward(self, x):
        out = self.cbr(x)
        return out
```

### 2. 데이터 전처리 및 증강

CIFAR-10 데이터셋에 대한 **데이터 전처리 및 증강**을 적용하여 모델의 성능을 향상시킵니다. 훈련 데이터에는 **랜덤 수평 반전**, **회전**, **아핀 변환**, **색상 조정** 등이 적용되며, 테스트 데이터는 정규화 후 텐서로 변환됩니다.

```python
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

### 3. 훈련 및 평가

- **훈련 함수**: `train()` 함수는 모델을 훈련하고, 각 배치마다 정확도를 계산합니다.
- **평가 함수**: `test()` 함수는 테스트 데이터에 대해 모델을 평가하고, 손실 값과 정확도를 계산합니다.
- **조기 종료**: 정확도가 목표를 달성하면 훈련을 자동으로 종료합니다.

```python
def train(model, optimizer, train_loader, loss_fn, device):
    total_labels = 0
    correct_labels = 0
    model.train()
    for images, labels in train_loader:
        labels = labels.to(device)
        images = images.to(device)
        
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        predictions = torch.max(outputs, 1)[1]
        total_labels += len(labels)
        correct_labels += (predictions == labels).sum()
    
    t_accuracy = correct_labels / total_labels
    return t_accuracy
```

### 4. 멀티-GPU 지원

`DataParallel`을 사용하여 여러 GPU에서 모델을 병렬로 학습할 수 있도록 지원합니다. 이를 통해 학습 속도를 크게 향상시킬 수 있습니다.

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

## 최종 목표

이 프로젝트의 목표는 **CIFAR-10** 데이터셋을 분류하는 **WideResNet** 모델을 학습시키고, 목표 정확도인 **75%**를 달성한 후 학습을 조기에 종료하는 것입니다. 이를 통해 **분산 학습** 및 **멀티-GPU 처리**에 대한 이해를 돕고, 실제 환경에서의 성능을 최적화하는 방법을 배울 수 있습니다.

---

## 프로젝트 구조

```
cifar10-wide-resnet/
│
├── assessment.py      # 모델 훈련 및 평가 코드
├── assessment_print.py  # 평가용 출력 함수 (수정하지 않음)
├── README.md           # 이 파일
└── data/               # CIFAR-10 데이터셋 (자동으로 다운로드)
```

## 참고 문헌

- [NVIDIA Deep Learning Institute (DLI)](https://www.nvidia.com/en-us/deep-learning-ai/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

---

이 리드미 파일은 프로젝트의 목적과 기능, 실행 방법을 간결하고 명확하게 설명하며, `WideResNet` 모델을 중심으로 분산 학습을 구현하는 방법을 설명합니다.
