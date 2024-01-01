
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.bn = torch.nn.BatchNorm2d(64)
 
    def forward(self, x1, x2):
        v1 = x1 + x2
        v2 = self.conv1(v1)
        v3 = v2 + x2
        v4 = self.conv2(v3)
        v5 = self.bn(v4)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
x2 = torch.randn(1, 64, 16, 16)
