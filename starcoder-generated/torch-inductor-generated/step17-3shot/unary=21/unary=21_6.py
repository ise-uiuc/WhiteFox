
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(5, 6, 7, stride=[-16383, -12345, -3141],
                                     dilation=[-843, -5, 1234567], padding=[314, 23, 78])
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm2d(64)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.relu(v1)
        v3 = self.bn(v2)
        return torch.tanh(v3)
# Inputs to the model
t = torch.randn(1, 5, 17, 23, 45)
