import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# from sklearn.cluster import KMeans
from tqdm import tqdm
from torch.utils.data import Subset

"""Dataset part"""

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)


mean_image = 0.0
total_samples = 0
denom = 0.0

for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc="Loading CIFAR-10")):
    batch_samples = inputs.size(0)
    mean_image += inputs.sum(dim=0)  # sum over batch dimension -> shape (C, H, W)
    total_samples += batch_samples

mean_image /= total_samples
mu_flat = mean_image.view(1, -1)

for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc="Computing denominator")):
    # inputs: shape (B, C, H, W)
    batch_flat = inputs.view(inputs.size(0), -1)
    
    # Compute squared distance to mean for each sample in batch
    dists_squared = ((batch_flat - mu_flat) ** 2).sum(dim=1) 
    
    # Sum
    denom += dists_squared.sum().item()

q_values = torch.empty(total_samples) # Our q(x)
start_idx = 0

for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc="Computing q(x) for all dataset indices")):
    batch_size = inputs.size(0)
    end_idx = start_idx + batch_size

    # Compute squared distances to the mean
    batch_flat = inputs.view(batch_size, -1)
    dists_squared = ((batch_flat - mu_flat) ** 2).sum(dim=1) 

    # Compute q(x)
    q_batch = 0.5 * (1 / total_samples) + 0.5 * (dists_squared / denom)  
    q_values[start_idx:end_idx] = q_batch

    start_idx = end_idx

# 1 / q(x)
sampling_probs = (1.0 / q_values)
sampling_probs /= sampling_probs.sum()  # normalize to sum to 1

m = 1000  # TODO Use the general way later
sample_indices = torch.multinomial(sampling_probs, num_samples=m, replacement=True)

coreset = Subset(train_dataset, sample_indices.tolist())
coreset_loader = torch.utils.data.DataLoader(coreset, batch_size=64, shuffle=False)

"""Training model part"""

# Use MPS if available (for Macs), otherwise fallback
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load ResNet18
model = resnet18(num_classes=10)
model = model.to(device)

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=300)

# Training loop
def train(model, train_loader, epochs=300):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loop.set_postfix(loss=running_loss/(total/inputs.size(0)), acc=100.*correct/total)

        scheduler.step()

# Validation loop
def validate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")

# Start training
if __name__ == '__main__':
    epochs = 300
    train(model, train_loader, epochs=epochs)
    validate(model, val_loader)