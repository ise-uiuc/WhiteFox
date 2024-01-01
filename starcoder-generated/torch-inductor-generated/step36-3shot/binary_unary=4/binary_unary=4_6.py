
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return torch.relu(v1 + self.other)

# Initializing the model
m = Model(other=torch.randn(8))

# Inputs to the model
x1 = torch.randn(1, 8)
