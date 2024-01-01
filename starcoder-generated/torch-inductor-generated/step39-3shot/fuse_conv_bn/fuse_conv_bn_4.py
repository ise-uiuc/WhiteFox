
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
    def forward(self, x0):
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x3 = self.conv3(x2)
        return x1, x3
# Inputs to the model
x = torch.randn(2, 3, 6, 6)
