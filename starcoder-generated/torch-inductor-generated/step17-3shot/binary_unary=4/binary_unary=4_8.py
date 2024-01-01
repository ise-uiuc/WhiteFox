
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100)
 
    def forward(self, x1):
        v4 = self.linear(x1)
        v4 = v4 + other
        v5 = F.relu(v4)
        return v5

# Initializing the model
m = Model(torch.tensor(0.7071067811865476))

# Inputs to the model
x1 = torch.randn(1, 100)
