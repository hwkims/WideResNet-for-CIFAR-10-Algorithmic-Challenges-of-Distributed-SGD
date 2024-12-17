import argparse
import torch
import torch.nn as nn
import time
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Argument parsing
parser = argparse.ArgumentParser(description='WideResNet Training with different batch sizes',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.01, help='learning rate for a single GPU')
parser.add_argument('--target-accuracy', type=float, default=.85, help='Target accuracy to stop training')
parser.add_argument('--patience', type=int, default=2, help='Number of epochs that meet target before stopping')

args = parser.parse_args()

# Define WideResNet model (same as in your code)
class cbrblock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(cbrblock, self).__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=(1,1), padding='same', bias=False), 
            nn.BatchNorm2d(output_channels), 
            nn.ReLU()
        )
    def forward(self, x):
        out = self.cbr(x)
        return out

class conv_block(nn.Module):
    def __init__(self, input_channels, output_channels, scale_input):
        super(conv_block, self).__init__()
        self.scale_input = scale_input
        if self.scale_input:
            self.scale = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=(1,1), padding='same')
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

class WideResNet(nn.Module):
    def __init__(self, num_classes):
        super(WideResNet, self).__init__()
        nChannels = [1, 16, 160, 320, 640]
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

def train(model, optimizer, train_loader, loss_fn, device):
    model.train()
    for images, labels in train_loader:
        labels = labels.to(device)
        images = images.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(model, test_loader, loss_fn, device):
    total_labels = 0
    correct_labels = 0
    loss_total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            labels = labels.to(device)
            images = images.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            predictions = torch.max(outputs, 1)[1]
            total_labels += len(labels)
            correct_labels += (predictions == labels).sum()
            loss_total += loss
    
    v_accuracy = correct_labels / total_labels
    v_loss = loss_total / len(test_loader)
    return v_accuracy, v_loss

# Main loop to run experiments with different batch sizes
batch_sizes = [16, 32, 64, 128]

for batch_size in batch_sizes:
    print(f"Training with batch size = {batch_size}")
    
    # Create dataset and dataloaders
    train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, drop_last=True)
    
    # Create model and optimizer
    num_classes = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = WideResNet(num_classes)
    
    # Use multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        model = nn.DataParallel(model)  # 모델을 여러 GPU에 복제하여 실행
    
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr)
    
    val_accuracy = []
    
    for epoch in range(args.epochs):
        t0 = time.time()  # 시작 시간 측정
        train(model, optimizer, train_loader, loss_fn, device)
        epoch_time = time.time() - t0  # 훈련 시간 측정
        v_accuracy, v_loss = test(model, test_loader, loss_fn, device)
        val_accuracy.append(v_accuracy)
        
        print(f"Epoch {epoch+1}/{args.epochs}: Epoch Time = {epoch_time:.3f}s, Validation Loss = {v_loss:.3f}, Validation Accuracy = {val_accuracy[-1]:.3f}")
        
        # Early stopping based on validation accuracy
        if val_accuracy[-1] >= args.target_accuracy:
            print(f"Early stopping after epoch {epoch + 1}")
            break
