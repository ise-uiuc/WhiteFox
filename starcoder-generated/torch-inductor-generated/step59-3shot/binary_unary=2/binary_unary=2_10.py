
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(6)
        self.bn2 = torch.nn.BatchNorm2d(10)
    def forward(self, x1):
        v1 = self.bn1(x1)
        v2 = v1 - x1
        v3 = self.bn2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
