
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.conv2 = torch.nn.Conv2d(3, 3, 3)
        self.bn2 = torch.nn.BatchNorm2d(3)
    def forward(self, x):
        x1 = 0
        x2 = self.conv1(x)
        x2 = self.bn1(x2)
        x3 = self.conv2(x2)
        x3 = self.bn2(x3)
        y = 0
# Inputs to the model
x = torch.randn(1, 3, 4, 4)
