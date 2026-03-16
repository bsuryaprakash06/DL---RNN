# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset
<img width="666" height="193" alt="image" src="https://github.com/user-attachments/assets/df2478f9-fd91-4db5-980e-b641ea4ac533" />


## DESIGN STEPS
### Step 1: Import Required Libraries

Import necessary Python libraries such as NumPy, Pandas, Matplotlib, PyTorch, and Scikit-learn modules required for data preprocessing, model creation, training, and visualization.

### Step 2: Load and Preprocess the Dataset

Load the training and testing datasets from CSV files. Extract the **closing price** column and normalize the values using MinMaxScaler to scale the data between 0 and 1.

### Step 3: Create Input Sequences

Generate sequences of past stock prices to be used as input for the model. A sequence length of **60** is used, meaning the model learns from the previous 60 days to predict the next day's price.

### Step 4: Build the RNN Model

Define a Recurrent Neural Network (RNN) using PyTorch with the following components:

* RNN layers for capturing temporal patterns
* A fully connected (Linear) layer for producing the final prediction.

### Step 5: Train the Model

Train the RNN model using the training dataset with Mean Squared Error (MSE) as the loss function and the Adam optimizer. The model updates its weights over multiple epochs to minimize prediction error.

### Step 6: Test and Visualize Predictions

Use the trained model to predict stock prices on the test dataset. Convert the normalized predictions back to original values and plot the **actual vs predicted stock prices** to evaluate model performance.


## PROGRAM

### Name: Surya Prakash B
### Register Number: 212224230281

```python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

## Step 1: Load and Preprocess Data
# Load training and test datasets
df_train = pd.read_csv('trainset.csv')
df_test = pd.read_csv('testset.csv')

df_train.head()

# Use closing prices
train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)

# Normalize the data based on training set only
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)

# Create sequences
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)


x_train.shape, y_train.shape, x_test.shape, y_test.shape

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# Create dataset and dataloader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

## Step 2: Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 64, num_layers = 2, output_size = 1):
      super(RNNModel, self).__init__()
      self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
      self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
      out, _ = self.rnn(x)
      out = self.fc(out[:,-1,:])
      return out

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

!pip install torchinfo

from torchinfo import summary

# input_size = (batch_size, seq_len, input_size)
summary(model, input_size=(64, 60, 1))

import torch.optim as optim

# Instantiate the model (Moved from EaAR46dm99qb to ensure model is defined)
model = RNNModel().to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

## Step 3: Train the Model
def train_model(model, train_loader, criterion, optimizer, epochs = 20):
  train_losses = []
  model.train()
  for epoch in range(epochs):
    total_loss = 0
    for x_batch, y_batch in train_loader:
      x_batch, y_batch = x_batch.to(device), y_batch.to(device)
      optimizer.zero_grad()
      outputs = model(x_batch)
      loss = criterion(outputs, y_batch)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")
  return train_losses

train_losses = train_model(model, train_loader, criterion, optimizer)
# Plot training loss
print('Name: Surya Prakash B')
print('Register Number: 212224230281')
plt.plot(train_losses, label = 'Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

## Step 4: Make Predictions on Test Set
model.eval()
with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()

# Inverse transform the predictions and actual values
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

# Plot the predictions vs actual prices
print('Name: Surya Prakash B')
print('Register Number: 212224230281')
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.show()
print(f'Predicted Price: {predicted_prices[-1]}')
print(f'Actual Price: {actual_prices[-1]}')
```

### OUTPUT

## Training Loss Over Epochs Plot
<img width="582" height="387" alt="image" src="https://github.com/user-attachments/assets/02930e94-90a2-4595-99bc-7d029416cc57" />
<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/5d6c65bb-941d-4c65-9d0d-c00c90f9264b" />

## True Stock Price, Predicted Stock Price vs time
<img width="861" height="36" alt="image" src="https://github.com/user-attachments/assets/cc6d1060-f162-47df-8ef2-7b96634f71b5" />
<img width="859" height="547" alt="image" src="https://github.com/user-attachments/assets/4e4a35d0-a1cb-4c01-a1df-9c9844aa7196" />

### Predictions
<img width="859" height="36" alt="image" src="https://github.com/user-attachments/assets/42fd523f-c740-4556-8e39-0053610046dc" />

## RESULT
The RNN model was successfully trained on the stock price dataset and used to predict future closing prices.  
The predicted prices were compared with the actual prices, and the training loss curve and prediction graph were generated successfully.
