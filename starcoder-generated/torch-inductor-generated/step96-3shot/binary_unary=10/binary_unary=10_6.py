
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Linear(8, 32)
        self.bn = torch.nn.BatchNorm2d(32)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + other
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 8)
