
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.conv2 = torch.nn.Conv2d(3, 3, 3)
        self.bn2 = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        s = self.conv1(x1)
        s = self.bn1(s)
        s = self.conv2(s)
        s = self.bn2(s)
        return s
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
