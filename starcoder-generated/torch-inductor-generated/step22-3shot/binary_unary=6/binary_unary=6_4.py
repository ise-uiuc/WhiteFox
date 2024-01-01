
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10, bias=False)
        self.bn = torch.nn.BatchNorm2d(8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 1
        v3 = F.relu(v2)
        v4 = v2
        v5 = v4 - 1
        v6 = F.relu(v5)
        v7 = v5
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
other = torch.randn(1, 10)
