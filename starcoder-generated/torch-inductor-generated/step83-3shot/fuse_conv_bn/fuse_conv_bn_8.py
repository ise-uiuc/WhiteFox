
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 2, 1)
        self.bn1 = torch.nn.BatchNorm2d(2)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = self.bn1(x2)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 4, 4)
