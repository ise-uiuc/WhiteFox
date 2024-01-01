
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 8, 1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 8, 1, padding=0)
        self.conv3 = torch.nn.Conv2d(32, 8, 1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 8, 1, padding=1)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.conv3(x3)
        v4 = self.conv4(x4)
        v5 = v1 + v2
        return v5
# Inputs to the model
x1 = torch.randn(1, 64, 80, 80)
x2 = torch.randn(1, 64, 80, 80)
x3 = torch.randn(1, 32, 80, 80)
x4 = torch.randn(1, 16, 80, 80)
