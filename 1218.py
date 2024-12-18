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

# Argument parsing for multi-node, multi-GPU, and other training settings
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

# Define convolutional block followed by batch normalization
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

# Basic residual block with skip connection
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

# WideResNet class definition
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

# Setup for Distributed Training (if using multi-node, multi-GPU)
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # set device to match the rank

def cleanup():
    dist.destroy_process_group()

# Training function
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
    return correct_labels / total_labels

# Testing function
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
    return correct_labels / total_labels, loss_total / len(test_loader)

# DataLoader setup
def get_dataloaders(batch_size):
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
    train_set = torchvision.datasets.CIFAR10("./data", download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10("./data", download=True, train=False, transform=transform_test)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, drop_last=True)
    
    return train_loader, test_loader

# Main function for training and evaluation
if __name__ == '__main__':
    # Setup Distributed Training (if applicable)
    if args.num_nodes > 1:
        setup(rank=args.node_id, world_size=args.num_nodes * args.num_gpus)  # Total world_size = num_nodes * num_gpus
    
    # Prepare data loaders
    train_loader, test_loader = get_dataloaders(args.batch_size)
    
    # Define model, loss function, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WideResNet(num_classes=10).to(device)
    
    if args.num_gpus > 1:
        model = nn.DataParallel(model)  # If using DataParallel (multi-GPU on one node)
    if args.num_nodes > 1:
        model = DDP(model, device_ids=[args.node_id])  # If using DDP (multi-node setup)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9)
    
    val_accuracy = []
    total_time = 0

    # Training loop
    for epoch in range(args.epochs):
        t0 = time.time()
        
        t_accuracy = train(model, optimizer, train_loader, loss_fn, device)
        epoch_time = time.time() - t0
        total_time += epoch_time
        images_per_sec = len(train_loader) * args.batch_size / epoch_time
        
        v_accuracy, v_loss = test(model, test_loader, loss_fn, device)
        val_accuracy.append(v_accuracy)
        
        # Print progress
        assessment_print(f"Epoch = {epoch + 1}: Cumulative Time = {total_time:.3f}, Epoch Time = {epoch_time:.3f}, Images/sec = {images_per_sec:.1f}, Training Accuracy = {t_accuracy:.3f}, Validation Loss = {v_loss:.3f}, Validation Accuracy = {v_accuracy:.3f}")
        
        # Early stopping logic
        if len(val_accuracy) >= args.patience and all(acc >= args.target_accuracy for acc in val_accuracy[-args.patience:]):
            assessment_print(f'Early stopping after epoch {epoch + 1}')
            break

    # Cleanup distributed setup
    if args.num_nodes > 1:
        cleanup()
