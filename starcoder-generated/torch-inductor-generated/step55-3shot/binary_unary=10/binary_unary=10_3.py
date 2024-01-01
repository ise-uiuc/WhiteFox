
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.transform = torch.nn.Linear(size, 1)
 
    def forward(self, x):
        x1, x2 = x # Split the current input tensor
        x1b = self.transform(x1) # Apply a linear transformation to the first part of the input tensor
        x2b = F.relu(x2) # Apply the ReLU activation function to the second part of the input tensor
        x3 = x1b + x2b # Add another tensor to x1b
        return x3

# Initializing the model
m = Model(10)

# Inputs to the model
x1 = torch.randn(5, 10)
x2 = torch.randn(5, 10)
