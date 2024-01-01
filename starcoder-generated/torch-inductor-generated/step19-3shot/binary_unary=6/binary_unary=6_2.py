
x2 = torch.randn(3, 1) # Initialize 'other'

class Model(torch.nn.Module):
    def __init__(self, x2):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.bias = torch.nn.Parameter(torch.randn(2))

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - x2
        v3 = relu(v2)
        return v3

# Initializing the model
m = Model(x2)

# Inputs to the model
x1 = torch.randn(2, 3)
