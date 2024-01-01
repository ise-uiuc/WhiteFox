
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential()
        self.conv1.add_module('a', torch.nn.Module())
        self.conv1.add_module('b', torch.nn.Module())
        self.conv1.add_module('c', torch.nn.Module())
        self.conv1.add_module('d', torch.nn.Conv2d(1, 8, 1, stride=1, padding=1))
    def forward(self, x1):
        v1 = self.conv1.a(x1)
        v2 = self.conv1.b(x1)
        v3 = self.conv1.c(x1)
        v4 = torch.concat([v1, v2, v3], axis=1)
        return self.conv1.d(v4)
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
