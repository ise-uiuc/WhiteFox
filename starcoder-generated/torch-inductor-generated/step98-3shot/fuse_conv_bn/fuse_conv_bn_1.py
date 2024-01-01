
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x2):
        x1 = self.conv1(x2)
        x2 = self.bn(x1)
        return x2
# Inputs to the model
x2 = torch.randn(1, 3, 10, 10)
