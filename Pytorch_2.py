# coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
import pandas as pd

# PacketDataset class to load and process packet data from a CSV file
class PacketDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        packet = self.data.iloc[idx, 0:1].values  # Modify this line based on your CSV file structure
        packet = self.transform(packet)
        return packet


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(p=0.25)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.dropout2(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x


def train(model, train_dataloader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss


def test(model, test_dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for batch in test_dataloader:
            batch = batch.to(device)
            output = model(batch)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return test_loss, correct / len(test_dataloader.dataset)


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and split the dataset
    dataset = PacketDataset(csv_file='path/to/your/csv/file.csv')  # Replace with the actual path to your CSV file
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    # Data loaders
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = Net().to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training settings
    num_epochs = 10

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        test_loss, test_accuracy = test(model, test_dataloader, criterion, device)
        print(f'Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')


if __name__ == '__main__':
    main()
