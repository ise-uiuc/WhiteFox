
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        v1 = self.bn(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
