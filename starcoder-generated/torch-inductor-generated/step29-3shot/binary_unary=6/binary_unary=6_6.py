
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1) # Apply a linear transformation to the input tensor
        v2 = v1 - 3 # Subtract 'other' from the output of the linear transformation
        v3 = F.relu(v2) # Apply the ReLU activation function to the result
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
