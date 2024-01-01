
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 32)
        self.bn = torch.nn.BatchNorm1d(32)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        z = self.bn(v1)
        return torch.cat([z], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 4)
