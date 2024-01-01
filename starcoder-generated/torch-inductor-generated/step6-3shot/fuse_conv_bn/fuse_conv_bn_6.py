
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(1)
        self.conv1 = torch.nn.Conv2d(2, 2, 3)
    def forward(self, x):
        return self.conv1(self.bn(x))
# Inputs to the model
x = torch.randn(1, 2, 5, 5)
