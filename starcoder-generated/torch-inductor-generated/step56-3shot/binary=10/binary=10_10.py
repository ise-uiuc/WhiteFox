
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(64) 
 
    def forward(self, x):
        v1 = self.bn(x)
        v2 = v1 + x
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 12, 12)
x2 = torch.randn(1, 64, 12, 12)
