
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=2, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.conv2d = torch.nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(3)
        self.conv3 = torch.nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(3)
    def forward(self, x):
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.conv2d(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 6, 6)
