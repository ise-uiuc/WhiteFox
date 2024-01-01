
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 1024)
 
    def forward(self, x1, x2, x3):
        v1 = self.linear(x1)
        v2 = v1 + x2
        v3 = torch.nn.functional.relu(v2)
        v4 = self.linear(x3)
        v5 = v4 + v3
        v6 = v5 + x3
        v7 = torch.nn.functional.relu(v6)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
x2 = torch.randn(1, 64)
x3 = torch.randn(1, 64)
