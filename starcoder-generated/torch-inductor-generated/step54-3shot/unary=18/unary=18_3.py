
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v3 = v1.max(axis=[2, 3], keepdims=True)[0]
        v2 = self.conv2(v3)
        v4 = v2.max(axis=[2, 3], keepdims=True)[0]
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
