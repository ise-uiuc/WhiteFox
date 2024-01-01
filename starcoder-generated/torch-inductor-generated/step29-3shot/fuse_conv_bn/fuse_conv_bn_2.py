
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 5, 2)
        self.bn1 = torch.nn.BatchNorm2d(5)
        self.conv2 = torch.nn.Conv2d(5, 5, 2)
        self.bn2 = torch.nn.BatchNorm2d(5)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = self.bn1(x2)
        x4 = self.conv2(x3)
        x5 = self.bn2(x4)
        return x5
# Inputs to the model
x1 = torch.randn(1, 5, 4, 4)
