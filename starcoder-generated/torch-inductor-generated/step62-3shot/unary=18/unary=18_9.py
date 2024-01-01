
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, groups=8, padding=3, stride=1)
        self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v2 = self.pool(v2)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(2, 16, 156, 156)
