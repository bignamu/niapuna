import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data_path = "prescription\\105505ATB_prescription_filled.csv"
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
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create the model, loss function and optimizer
model = DNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 500
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

# Calculate the mean squared error for each output
mse = ((predictions_np - targets_np) ** 2).mean(axis=0)

# Plot the mean squared errors
plt.figure(figsize=(10, 6))
plt.bar(["1회 투약량", "1일투약량", "총투여일수", "단가", "금액"], mse)
plt.title("Mean Squared Error for Each Output")
plt.ylabel("MSE")
plt.show()
