
class Model(torch.nn.Module):
    def __init__(self, other: float):
        super().__init__()
        self.linear = torch.nn.Linear(3, 12)
        self.other = other
 
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = v1 - self.other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model(0.6)

# Inputs to the model
x2 = torch.randn(1, 3)
