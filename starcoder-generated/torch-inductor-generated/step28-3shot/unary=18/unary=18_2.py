
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7,stride=7)
        self.conv2 = torch.nn.Conv2d(64, 64, 7,stride=7)
        self.conv3 = torch.nn.Conv2d(64, 64, 7, stride=7)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
