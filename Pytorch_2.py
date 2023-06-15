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
    
    
    
    
 














import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KernelDensity
import pandas as pd
import numpy as np

# Define the dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx],dtype=torch.float32)
        y = torch.tensor(self.targets[idx])
        return x, y

# Define the Prototypical Network model
class PrototypicalNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(PrototypicalNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x, mask=None):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1)  # Apply mask to zero out padded elements
        return x


# Training Algorithm
def train(model, dataloader, Nepisodes):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for episode in range(1, Nepisodes+1):
        model.train()
        optimizer.zero_grad()

        support_examples = []
        for class_label in torch.unique(ytrain):
            class_examples = Xtrain[ytrain == class_label]
            support_indices = torch.randperm(class_examples.shape[0])[:S]

            # Create tensors of equal size for support examples
            max_length = max([examples.shape[0] for examples in class_examples])
            equal_size_examples = torch.zeros((S, max_length, class_examples.shape[1]))

            # Fill tensors with support examples
            for i, index in enumerate(support_indices):
                equal_size_examples[i, :class_examples[index].shape[0], :] = torch.tensor(class_examples[index],dtype=torch.float32)

            support_examples.append(equal_size_examples)
        print(support_examples)
        prototypes = torch.mean(model(torch.cat(support_examples, dim=0)), dim=1)

        for x, y in dataloader:
            embeddings = model(x)
            logits = -torch.cdist(embeddings, prototypes, p=2)
            class_probs = nn.functional.softmax(logits, dim=1)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    return model


# Load the dataset
data = pd.read_csv(r'E:\http_new_comx_analysis.csv')
df = data.drop(['Flow'], axis=1)

# Apply LabelEncoder to encode categorical labels
label_encoder = LabelEncoder()
df_encoded = df.apply(label_encoder.fit_transform)

X = df_encoded.drop(['Labels'], axis=1).values
y = df['Labels'].values

# Reshape y to be 1-dimensional
# y = y.reshape(-1)

# Split the dataset into train, calibration, and test sets
Xtrain, Xcal, Xtest = X[:800], X[800:900], X[900:]
ytrain, ycal, ytest = y[:800], y[800:900], y[900:]

# Convert ytrain to PyTorch tensor
ytrain = torch.tensor(ytrain, dtype=torch.long)

# Define the hyperparameters
input_dim = Xtrain.shape[1]
embedding_dim = 64
hidden_dim = 64
S = 5  # Number of support examples per class

# Create dataloaders for training, calibration, and test
train_dataset = TimeSeriesDataset(Xtrain, ytrain)
# cal_dataset = TimeSeriesDataset(Xcal, ycal)
test_dataset = TimeSeriesDataset(Xtest, ytest)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# cal_dataloader = DataLoader(cal_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Create and train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PrototypicalNetwork(input_dim, embedding_dim, hidden_dim)
model = model.to(device)
model = train(model, train_dataloader, Nepisodes=20000)

# Calibrate the model
# model = model.to('cpu')  # Move model back to CPU for calibration
# calibrated_models = calibrate(model, cal_dataloader)

    
