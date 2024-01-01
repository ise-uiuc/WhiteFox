
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, (3, 3))
        self.bn = torch.nn.BatchNorm2d(64)
 
    def forward(self, x):
        a1 = self.bn(self.conv(x))
        n1 = torch.flatten(a1,1)
        a2 = self.linear(n1)
        v1 = torch.transpose(a2, 0, 1)
        v2 = a2 - n1
        return n1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
