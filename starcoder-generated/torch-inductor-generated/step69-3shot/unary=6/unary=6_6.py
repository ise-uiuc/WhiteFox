
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.s = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.z = torch.nn.BatchNorm2d(3)
        self.c = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.b = torch.nn.BatchNorm2d(3)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        v1 = self.c(self.z(self.s(x1)))
        v2 = self.b(v1)
        v3 = torch.nn.functional.relu(v2)
        v4 = v2 + v3
        v5 = v4.repeat(4, 1, 2, 2)
        return self.bn(v5)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
