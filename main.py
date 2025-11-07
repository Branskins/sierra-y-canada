import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2

# Get cpu, gpu or mps device for training.
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def main():
    train_transform = v2.Compose([
        v2.Resize(size=(256,256)),
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize(
        #     mean=[85.6542, 80.4268, 72.8841],
        #     std=[93.3963, 88.0354, 82.0991]
        # )
    ])
    test_transform = v2.Compose([
        v2.Resize(size=(256,256)),
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize(
        #     mean=[231.0843, 225.8617, 219.1473],
        #     std=[43.8241, 48.9563, 60.1995]
        # )
    ])

    train_data = PokemonDataset('data/train_labels.csv', 'data/train', train_transform)
    test_data = PokemonDataset('data/test_labels.csv', 'data/test', test_transform)

    batch_size = 8
    train_dataloader = DataLoader(train_data, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size, shuffle=True)

    model = NeuralNetwork(196608).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-3)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")


# Build custom dataset
class PokemonDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, dtype={'label': np.int64})
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path, mode='RGB')
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
# Build model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.sequential_stack = nn.Sequential(
            nn.Linear(input_size, 6144),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(6144, 3072),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(3072, 1536),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1536, 768),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(384, 151)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.sequential_stack(x)
        return logits
    
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Resets gradients
        optimizer.zero_grad()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch % 15 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error:\n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>7f} \n")

if __name__ == '__main__':
    main()
