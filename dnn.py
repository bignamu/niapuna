import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


# Load the data
data_path = "480600ATB_prescription_filled.csv"
data = pd.read_csv(data_path)

# Fill missing values with the median
for column in ["1회 투약량", "1일투약량"]:
    data[column].fillna(data[column].median(), inplace=True)

# Remove outliers using the IQR method
for column in ["1회 투약량", "1일투약량", "총투여일수", "단가", "금액"]:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR)))]

# Select the features and the target
features = data[["성별코드", "연령대코드(5세단위)"]]
target = data[["1회 투약량", "1일투약량", "총투여일수", "단가", "금액"]]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Create a TensorDataset from the tensors
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create a DataLoader from the TensorDataset
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the DNN model
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(2, 128)  # Increase the number of neurons
        self.fc2 = nn.Linear(128, 128)  # Increase the number of neurons
        self.dropout = nn.Dropout(0.5)  # Add dropout layer
        self.fc3 = nn.Linear(128, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after activation
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout after activation
        x = self.fc3(x)
        return x


# Create the model, loss function and optimizer
model = DNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_loader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)

# Convert the predictions and the targets to numpy arrays
predictions_np = predictions.numpy()
targets_np = y_test_tensor.numpy()

# 1. Feature distribution visualization
for column in features.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(features[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.show()

# 2. Target and prediction distribution visualization
for i in range(predictions_np.shape[1]):
    plt.figure(figsize=(6, 4))
    sns.kdeplot(targets_np[:, i], label='Targets', shade=True)
    sns.kdeplot(predictions_np[:, i], label='Predictions', shade=True)
    plt.title(f"Distribution of Targets and Predictions for {target.columns[i]}")
    plt.legend()
    plt.show()

# 3. Scatter plot of targets and predictions
for i in range(predictions_np.shape[1]):
    plt.figure(figsize=(6, 6))
    plt.scatter(targets_np[:, i], predictions_np[:, i], alpha=0.3)
    plt.xlabel('Targets')
    plt.ylabel('Predictions')
    plt.title(f"Scatter Plot of Targets vs Predictions for {target.columns[i]}")
    plt.show()

# 4. Error plot of targets and predictions
for i in range(predictions_np.shape[1]):
    plt.figure(figsize=(6, 4))
    sns.histplot(targets_np[:, i] - predictions_np[:, i], kde=True)
    plt.title(f"Error Distribution for {target.columns[i]}")
    plt.show()

# 5. Heatmap of targets and predictions
correlation_matrix = np.corrcoef(targets_np.T, predictions_np.T)
plt.figure(figsize=(8, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm",
            xticklabels=target.columns, yticklabels=target.columns)
plt.title("Correlation Heatmap of Targets and Predictions")
plt.show()


# Set the size of subplots
fig, axs = plt.subplots(5, 2, figsize=(20, 30))

# 1. Feature distribution visualization
for i, column in enumerate(features.columns):
    sns.histplot(features[column], kde=True, ax=axs[i, 0])
    axs[i, 0].set_title(f"Distribution of {column}")

# 2. Target and prediction distribution visualization
for i in range(predictions_np.shape[1]):
    sns.kdeplot(targets_np[:, i], label='Targets', shade=True, ax=axs[i, 1])
    sns.kdeplot(predictions_np[:, i], label='Predictions', shade=True, ax=axs[i, 1])
    axs[i, 1].set_title(f"Distribution of Targets and Predictions for {target.columns[i]}")
    axs[i, 1].legend()

plt.tight_layout()
plt.show()