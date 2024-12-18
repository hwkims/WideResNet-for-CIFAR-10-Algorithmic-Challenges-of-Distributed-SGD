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

이 코드는 분산 학습을 사용하여 CIFAR-10 데이터셋에 대해 **WideResNet** 모델을 훈련시키는 PyTorch 코드입니다. 각 줄의 주요 역할에 대해 설명하겠습니다.

---

### 1-6: Importing Libraries
```python
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
from torch.utils.data import DataLoader
from assessment_print import assessment_print
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
```
필요한 라이브러리들을 임포트합니다.
- `argparse`: 명령줄 인자 파싱을 위한 라이브러리.
- `torch`와 `torch.nn`: PyTorch의 기본 라이브러리 및 신경망 모듈.
- `torchvision`: CIFAR-10 데이터셋과 모델을 다루는 라이브러리.
- `transforms`: 이미지 전처리를 위한 라이브러리.
- `time`: 시간을 측정하는 라이브러리.
- `DataLoader`: 데이터를 배치 단위로 로딩하는 기능.
- `assessment_print`: 맞춤형 출력 함수 (예를 들어 학습 진행 상황 출력).
- `torch.distributed`: 분산 학습을 위한 모듈.
- `DistributedDataParallel`: 분산 GPU 학습을 위한 DDP 모델.

---

### 8-15: Argument Parsing
```python
parser = argparse.ArgumentParser(description='Workshop Assessment',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--node-id', type=int, default=0, help='Node ID for distributed training')
parser.add_argument('--num-gpus', type=int, default=4, help='Number of GPUs to use')
parser.add_argument('--num-nodes', type=int, default=1, help='Number of nodes to use')
parser.add_argument('--batch-size', type=int, default=128, help='Input batch size for training')
parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.01, help='Learning rate for a single GPU')
parser.add_argument('--target-accuracy', type=float, default=0.75, help='Target accuracy to stop training')
parser.add_argument('--patience', type=int, default=2, help='Number of epochs that meet target before stopping')
args = parser.parse_args()
```
명령줄 인자를 처리합니다. 이를 통해 사용자로부터 여러 파라미터를 받을 수 있습니다. 예를 들어, `--batch-size`로 배치 크기를 지정하거나, `--epochs`로 훈련할 에폭 수를 지정할 수 있습니다.

---

### 17-27: Convolutional Block (`cbrblock`)
```python
class cbrblock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(cbrblock, self).__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=(1, 1), padding='same', bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.cbr(x)
```
- `cbrblock` 클래스는 **Conv2D** -> **BatchNorm** -> **ReLU** 순서로 구성된 기본적인 합성곱 블록을 정의합니다.
- `Conv2d`는 3x3 크기의 커널을 사용하여 이미지를 처리합니다.
- `BatchNorm2d`는 배치 정규화를 적용하여 학습 안정성을 높입니다.
- `ReLU`는 활성화 함수로 비선형성을 추가합니다.

---

### 29-39: Residual Block (`conv_block`)
```python
class conv_block(nn.Module):
    def __init__(self, input_channels, output_channels, scale_input):
        super(conv_block, self).__init__()
        self.scale_input = scale_input
        if self.scale_input:
            self.scale = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=(1, 1), padding='same')
        self.layer1 = cbrblock(input_channels, output_channels)
        self.dropout = nn.Dropout(p=0.01)
        self.layer2 = cbrblock(output_channels, output_channels)
        
    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.dropout(out)
        out = self.layer2(out)
        if self.scale_input:
            residual = self.scale(residual)
        out = out + residual
        return out
```
- `conv_block` 클래스는 **Residual Block**을 정의합니다. 입력과 출력을 더하는 **skip connection**을 포함한 블록입니다.
- 두 개의 `cbrblock`을 거쳐서 출력을 만들고, `scale_input`이 참이면 입력 채널을 출력 채널 크기에 맞추는 **1x1 Convolution**을 추가합니다.
- 마지막에 입력값 `x`와 출력을 더하여 **Residual Learning**을 구현합니다.

---

### 41-60: WideResNet Model
```python
class WideResNet(nn.Module):
    def __init__(self, num_classes):
        super(WideResNet, self).__init__()
        nChannels = [3, 16, 160, 320, 640]
        self.input_block = cbrblock(nChannels[0], nChannels[1])
        self.block1 = conv_block(nChannels[1], nChannels[2], 1)
        self.block2 = conv_block(nChannels[2], nChannels[2], 0)
        self.pool1 = nn.MaxPool2d(2)
        self.block3 = conv_block(nChannels[2], nChannels[3], 1)
        self.block4 = conv_block(nChannels[3], nChannels[3], 0)
        self.pool2 = nn.MaxPool2d(2)
        self.block5 = conv_block(nChannels[3], nChannels[4], 1)
        self.block6 = conv_block(nChannels[4], nChannels[4], 0)
        self.pool = nn.AvgPool2d(7)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(nChannels[4], num_classes)

    def forward(self, x):
        out = self.input_block(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.pool1(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.pool2(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out
```
- `WideResNet`은 **Wide Residual Network**의 구현입니다.
- 여러 `conv_block`과 **MaxPooling** 및 **Average Pooling**을 사용하여 이미지의 특징을 추출합니다.
- 최종적으로 **Fully Connected Layer (fc)**를 사용하여 클래스 수 만큼 출력합니다.

---

### 62-65: Distributed Setup
```python
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # set device to match the rank
```
- **분산 학습**을 위한 초기화 함수입니다. `rank`와 `world_size`를 이용해 각 노드의 GPU가 올바르게 설정되도록 합니다.
- `nccl` backend는 NVIDIA GPU 간 통신을 최적화한 라이브러리입니다.

---

### 67-70: Cleanup
```python
def cleanup():
    dist.destroy_process_group()
```
- 분산 학습을 마친 후 리소스를 정리하는 함수입니다.

---

### 73-88: Training and Testing Functions
```python
def train(model, optimizer, train_loader, loss_fn, device):
    # Training logic
    ...
def test(model, test_loader, loss_fn, device):
    # Testing logic
    ...
```
- `train`: 모델을 훈련하는 함수입니다. 주어진 데이터를 사용하여 모델을 학습하고, 학습된 모델로 예측을 수행한 후 정확도를 반환합니다.
- `test`: 모델을 평가하는 함수입니다. 검증 데이터셋에 대해 모델을 평가하고, 정확도와 손실을 반환합니다.

---

### 90-104: DataLoader Setup
```python
def get_dataloaders(batch_size):
    # 데이터 로딩 함수
    ...
```
- CIFAR-10 데이터셋을 훈련 및 테스트 세트로 나누어 **DataLoader**로 반환합니다.
- 데이터 증강 기법(`RandomHorizontalFlip`, `RandomRotation`, `RandomAffine`, `ColorJitter`)을 포함한 `transform_train`과, 정규화된 `transform_test`를 설정합니다.

---

### 106-132: Main Function
```python
if __name__ == '__main__':
    # Main function for training and evaluation
    ...
```
- **훈련**과 **평가**를 위한 메인 함수입니다.
- `setup`: 분산 학습을 설정합니다 (다중 노드 및 다중 GPU).
- 모델과 최적화 기법, 손실 함수 설정.
- 훈련 및 테스트 루프를 통해 모델을 학습하고 평가합니다.
- `early stopping` 조건을 만족하면 학습을 중지합니다.

---

### 134-136: Cleanup After Training
```python
if args.num_nodes > 1:
    cleanup()
```
- 분산 학습을 사용한 경우, 학습 후 **process group**을 정리합니다.

---

이 코드는 분산 학습을 통해 CIFAR-10 데이터셋을 **WideResNet** 모델로 훈련하고 평가하는 과정을 구현한 것입니다.
