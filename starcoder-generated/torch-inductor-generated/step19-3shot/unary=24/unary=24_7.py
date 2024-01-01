
class Model(torch.nn.Module):
    def __init__(self, channels=8, input_shape=(64, 64)):
        super().__init__()
        width, height = input_shape
        self.conv = torch.nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(channels)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.bn(v1)
        v3 = v2.clone()
        v4 = torch.where(v2 > 1.0, v2, torch.zeros_like(v2))
        return v4
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
