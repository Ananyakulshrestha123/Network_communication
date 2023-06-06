import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
import pandas as pd

# Define the dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
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
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
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
            support_examples.append(class_examples[support_indices])

        support_examples = torch.stack(support_examples)
        prototypes = torch.mean(model(support_examples), dim=1)

        for x, y in dataloader:
            embeddings = model(x)
            logits = -torch.cdist(embeddings, prototypes, p=2)
            class_probs = nn.functional.softmax(logits, dim=1)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    return model

# Calibrate Algorithm
def calibrate(model, dataloader):
    model.eval()
    embeddings = []
    targets = []

    with torch.no_grad():
        for x, y in dataloader:
            embeddings.append(model(x))
            targets.append(y)

    embeddings = torch.cat(embeddings, dim=0)
    targets = torch.cat(targets, dim=0)

    prototypes = []
    for class_label in torch.unique(targets):
        class_embeddings = embeddings[targets == class_label]
        prototype = torch.mean(class_embeddings, dim=0)
        prototypes.append(prototype)

    distances = torch.cdist(embeddings, torch.stack(prototypes), p=2)
    relative_distances = distances / torch.std(distances)

    calibrated_models = []
    for class_label in torch.unique(targets):
        class_distances = relative_distances[targets == class_label].detach().numpy()
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(class_distances.reshape(-1, 1))
        calibrated_models.append(kde)

    return calibrated_models

# Test Algorithm
def test(model, calibrated_models, dataloader):
    model.eval()
    embeddings = []
    targets = []

    with torch.no_grad():
        for x, y in dataloader:
            embeddings.append(model(x))
            targets.append(y)

    embeddings = torch.cat(embeddings, dim=0)
    targets = torch.cat(targets, dim=0)

    logits = -torch.cdist(embeddings, torch.stack(prototypes), p=2)
    class_probs = nn.functional.softmax(logits, dim=1)

    distances = torch.cdist(embeddings, torch.stack(prototypes), p=2)
    relative_distances = distances / torch.std(distances)

    p_values = torch.zeros(embeddings.shape[0])
    for i, class_label in enumerate(torch.unique(targets)):
        class_distances = relative_distances[targets == class_label].detach().numpy()
        p_values += 1 - calibrated_models[i].score_samples(class_distances.reshape(-1, 1))

    confidence = 1 - p_values
    return confidence

# Load and preprocess the dataset
data = pd.read_csv(r'E:\reliance_historical_prices.csv')
print(data)
X = data.drop('Trades',axis=1).values
y = data['Trades'].values

# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Split the data into train, calibration, and test sets
Xtrain = X[:800]
ytrain = y[:800]
Xcal = X[800:1000]
ycal = y[800:1000]
Xtest = X[1000:]
ytest = y[1000:]

# Create dataloaders for training and calibration
train_dataset = TimeSeriesDataset(Xtrain, ytrain)
cal_dataset = TimeSeriesDataset(Xcal, ycal)
test_dataset = TimeSeriesDataset(Xtest, ytest)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
cal_dataloader = DataLoader(cal_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Create and train the Prototypical Network model
model = PrototypicalNetwork(input_dim=Xtrain.shape[1], embedding_dim=64, hidden_dim=64)
model = train(model, train_dataloader, Nepisodes=10)

# Calibrate the model
calibrated_models = calibrate(model, cal_dataloader)

# Test the model
confidence = test(model, calibrated_models, test_dataloader)

print(confidence)






calibrate(model, dataloader):

# This function performs calibration of the trained model.
# It takes the trained model and a dataloader as input for the calibration data.
# The function sets the model to evaluation mode (model.eval()) and initializes empty lists for storing embeddings and targets.
# It then iterates over the calibration dataloader and computes embeddings for each input sample, storing them along with their respective targets.
# After processing all calibration samples, the embeddings and targets are concatenated into tensors.
# The prototypes for each class are calculated by taking the mean of the class embeddings.
# The distances between the embeddings and prototypes are computed and normalized by dividing by the standard deviation of the distances.
# Calibrated models are constructed for each class by fitting a Gaussian kernel density estimator (KernelDensity) to the normalized distances.
# The calibrated models are returned as a list.
# test(model, calibrated_models, dataloader):

# This function tests the calibrated model on the test data.
# It takes the model, calibrated_models (list of calibrated models), and a dataloader for the test data as inputs.
# Similar to the calibration step, the function sets the model to evaluation mode, computes embeddings for the test data, and stores them along with their targets.
# The logits (distances) between the embeddings and prototypes are calculated.
# The distances are normalized by dividing by the standard deviation of the distances.
# P-values are computed by subtracting the scores obtained from the calibrated models for each class.
# Confidence scores are calculated as 1 minus the p-values.
# The confidence scores are returned as a tensor.
# Dataset preparation and data splitting:

# The code loads and preprocesses a time series dataset from a CSV file.
# The input features (X) are extracted by dropping the 'Trades' column, and the corresponding labels (y) are extracted.
# The dataset is split into three parts: training (first 800 samples), calibration (800-1000 samples), and testing (from the 1000th sample onwards).
# Datasets and dataloaders are created for training, calibration, and testing.
# Model initialization, training, and
