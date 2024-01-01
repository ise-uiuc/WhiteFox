
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 2)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(2, 3, 10, 10)
