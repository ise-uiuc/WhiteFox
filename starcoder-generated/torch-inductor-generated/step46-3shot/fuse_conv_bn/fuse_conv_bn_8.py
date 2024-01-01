
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(100, 16, 5, stride=1, padding=1, bias=False)
        torch.manual_seed(50)
        self.bn1 = torch.nn.BatchNorm2d(1)
        torch.manual_seed(50)
        self.bn2 = torch.nn.BatchNorm2d(16)
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = y*y
        y = self.bn2(y)
        return y
# Inputs to the model
x = torch.randn(1, 100, 30, 30)
