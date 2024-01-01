
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c2 = torch.nn.Conv2d(3, 16, 3)
        self.bn = torch.nn.BatchNorm2d(16)
    def forward(self, x1):
        return self.bn(self.c2(x1))
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
