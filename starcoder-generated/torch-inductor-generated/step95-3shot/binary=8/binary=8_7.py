
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(32, 1e-05, 0.1, True)
        self.conv2 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(32, 1e-05, 0.1, True)
        self.conv3 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0, bias=False)
    def forward(self, x):
        v1 = self.bn1(self.conv1(x))
        v2 = self.bn2(self.conv2(x))
        v3 = v1 + v2
        v4 = self.conv3(v3)
        return v4
# Inputs to the model
x = torch.randn(5, 3, 128, 128)
