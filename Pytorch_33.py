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
