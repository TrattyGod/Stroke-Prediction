import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

class Model(nn.Module):
    def __init__(self, in_features=16, h1=32, h2=24, h3=12, h4=6, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.out = nn.Linear(h4, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.sigmoid((self.out(x)))
        return x

torch.manual_seed(41)
model = Model()

my_df = pd.read_csv("DATA.csv")

pd.set_option('future.no_silent_downcasting', True)

# This is training data, so we must drop the answers
X = my_df.drop("At Risk (Binary)", axis=1).values
y = my_df["At Risk (Binary)"].values

# Ensure correct data types
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32) 
y_test = torch.tensor(y_test, dtype=torch.float32)    

#y_train = torch.LongTensor(y_train)
#y_test = torch.LongTensor(y_test)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 260
losses = []

for i in range(epochs):
    y_pred = model.forward(X_train).squeeze()

    loss = criterion(y_pred, y_train)
    losses.append(loss.detach().numpy())

    if (i % 10 == 0):
        print(f"Epoch: {i}, Loss: {loss}")

    # backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "MODEL.pt")

# #plt.plot(range(epochs), losses)
# #plt.ylabel("Loss/error")
# #plt.xlabel("Epoch")
# #plt.show()

loaded_model = Model()
loaded_model.load_state_dict(torch.load("MODEL.pt"))

correct = 0
amount = 0

with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = loaded_model.forward(data.unsqueeze(0)).item()
        y_val_binary = 1 if y_val > 0.5 else 0

        print(f"{i+1} - {str(y_val_binary)}, - {y_test[i]}")

        if (y_val_binary == y_test[i]):
            correct += 1
        
        amount += 1

print(f"ACCURACY: {correct}/{amount}, {(correct/amount)*100}%")