
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 1000)
 
    def forward(self, x1, x2, x3, x4, x5):
        v1 = self.linear(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        v4 = self.linear(x3)
        v5 = v4 + x4
        v6 = torch.relu(v5)
        v7 = self.linear(x5)
        v8 = v7 + x5
        v9 = torch.relu(v8)
        return torch.cat([v3, v6, v9])

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
x2 = torch.randn(2, 5)
x3 = torch.randn(3, 5)
x4 = torch.randn(4, 5)
x5 = torch.randn(5, 5)
