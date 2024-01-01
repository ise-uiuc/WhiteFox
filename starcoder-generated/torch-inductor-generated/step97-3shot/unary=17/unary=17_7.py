
import torch
def model(x1):
    v0 = torch.reshape(x1, shape=(1, 16, 28, 28)) # Reshape input tensor to (1, 16, 28, 28)
    v1 = torch.unsqueeze(v0, dim=1) # Unsqueeze the shape of tensor v0 by inserting a dimension of 1 at the position of 1.
    v2 = torch.conv1d(v1, weight=torch.unsqueeze(torch.arange(0,16, dtype=torch.float), dim=1), stride=2, padding=1, dilation=1) # Performs the 1D convolution.
    v3 = torch.relu(v2) # Apply the ReLU function to the result of the previous pointwise convolution.
    v4 = torch.squeeze(v3, dim=1) # Squeeze a dimension of the shape of tensor v3 by removing the dimension of size 1 at the position of 1.
    v5 = torch.conv1d(v4, weight=torch.unsqueeze(torch.arange(0,16, dtype=torch.float), dim=1), stride=2, padding=1, dilation=1) # Performs the 1D convolution.
    v6 = torch.relu(v5) # Apply the ReLU function to the result of the previous pointwise convolution.
    v7 = torch.sigmoid(v6) # Apply pointwise scalar multiplication to the result of the previous pointwise convolution.
    v8 = torch.reshape(v7, shape=(1, 16, 14, 14)) # Reshape the output tensor v7 to (1, 16, 14, 14).
    v9 = torch.shape(v8, out=None) # Returns a tuple (torch.Size) of sizes of each dimension.
    v10 = torch.reshape(v8, shape=v9 ) # Reshape tensor v8 to the shape specified in v9.
    return v10
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
