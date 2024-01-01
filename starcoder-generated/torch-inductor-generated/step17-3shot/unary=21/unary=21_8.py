
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(77, 109, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(97, 82, 21, stride=18, padding=12, dilation=14)
        self.conv3 = torch.nn.Conv3d(8, 40, 7, stride=3, dilation=10, groups=2)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.relu(v1)
        v3 = self.conv2(v2)
        v4 = self.relu(v3)
        v5 = self.conv3(v4)
        v6 = torch.tanh(v5)
        return v6
# Inputs to the model
x = torch.randn(6, 8, 56, 56, 10)
