
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(7, 7, 2)
        self.bn1 = torch.nn.BatchNorm2d(7)
        self.conv2 = torch.nn.Conv2d(7, 7, 2)
        self.bn2 = torch.nn.BatchNorm2d(7)
        self.conv3 = torch.nn.Conv2d(7, 3, 1)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.bn2(x2)
        x4 = self.bn2(x1)
        x5 = self.conv3(x4)
        return x5
# Inputs to the model
x = torch.randn(1, 7, 4, 4)
