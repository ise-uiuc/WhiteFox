
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = torch.nn.Linear(3, 8) # Apply linear transformation to the input tensor

    def forward(self, x1):
        v1 = torch.tanh(self.m1(x1)) # Apply hyperbolic tangent function to the output of the linear transformation
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 3)
