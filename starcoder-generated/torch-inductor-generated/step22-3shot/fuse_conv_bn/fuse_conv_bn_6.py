
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(4)
    def forward(self, x1):
        return self.bn(x1)
# Inputs to the model
x1 = torch.randn(1, 4, 4, 4)
