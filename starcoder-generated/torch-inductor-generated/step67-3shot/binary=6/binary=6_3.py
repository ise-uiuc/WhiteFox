
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(8)
        self.linear = torch.nn.Linear(8, 3)
 
    def forward(self, x1):
        v1 = self.bn(x1)
        v2 = v1.transpose(1, -1)
        v3 = self.linear(v2)
        v4 = v3 - 1
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 32, 32)
other = torch.randn(1, )
