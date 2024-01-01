
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(5)
    def forward(self, x4):
        x1 = self.bn(x4)
        x2 = self.bn(x1)
        x3 = self.bn(x1)
        out = (x1, x2, x3)
        return out
# Inputs to the model
x4 = torch.randn(2, 5, 4, 4)
