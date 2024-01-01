
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 5, 2)
        self.bn1 = torch.nn.BatchNorm2d(5)
        self.conv2 = torch.nn.Conv2d(5, 3, 1)
    def forward(self, x2):
        x1 = self.conv1(x2)
        x3 = self.bn1(x1)
        x4 = self.conv2(x3)
        return x4
# Inputs to the model
x2 = torch.randn(1, 5, 4, 4)
