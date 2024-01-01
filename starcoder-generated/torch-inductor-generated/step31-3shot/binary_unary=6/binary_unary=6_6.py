
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.other = other
    
    def forward(self, x1):
        x2 = self.linear(x1)
        x3 = x2 - self.other
        x4 = torch.relu(x3)
        return x4

# Initializing the model
m = Model(10.0)

# Inputs to the model
x1 = torch.randn(32, 1)
