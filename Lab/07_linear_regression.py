import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
# Step - 0: data prepartion
# step -1 : Model
# step - 2: loss and Optimizer
# step - 3: traning loop

# 0) Prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

# convert (cast) to float Tensor
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
# rephase... one column
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# 1) Model
# Linear model f = wx + b. Here one layer .. use built-in linear model .. it needs input size of feature and output size
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)
# set up the model .. in case of liner regression MSE (nn.MSELoss()) and set the optimizer (SGD).. this SGD requires model parameter and learning rate
# 2) Loss and optimizer.
learning_rate = 0.01

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# 3) Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass and loss,
    y_predicted = model(X)
    loss = criterion(y_predicted, y)
    
    # Backward pass and update
    loss.backward()
    optimizer.step()
    #update
	# empty our graidetns .. v.v imp
    # zero grad before new step
    optimizer.zero_grad()
	# print some information
    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# Plot
predicted = model(X).detach().numpy()

plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
