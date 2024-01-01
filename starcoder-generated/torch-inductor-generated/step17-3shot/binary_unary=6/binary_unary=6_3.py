
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 32)
 
    def forward(self, x1, x2, x3, x4, x5):
        v1 = self.linear(x1)
        v2 = v1 - x2
        v3 = F.relu(v2)
        v4 = v3 - x3
        v5 = F.relu(v4)
        v6 = v5 - x4
        v7 = F.relu(v6)
        v8 = v7 - x5
        v9 = F.relu(v8)
        return v9

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
x2 = torch.randn(1, 2)
x3 = torch.randn(1, 4)
x4 = torch.randn(1, 1)
x5 = torch.randn(1, 9)
