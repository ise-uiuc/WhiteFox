
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 10, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 10, 3, stride=2, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(10)
        self.layer_norm1 = torch.nn.LayerNorm([512, 512], 1e-05, 0.1)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.conv3(x3)
        v4 = v1 + v2 + v3
        v5 = self.bn1(v4)
        v6 = self.layer_norm1(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 32, 32)
x3 = torch.randn(1, 3, 16, 16)
