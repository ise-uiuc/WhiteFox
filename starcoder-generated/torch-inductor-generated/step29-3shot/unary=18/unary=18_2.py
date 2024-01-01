
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.dp1 = torch.nn.Dropout(p = 0.0)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.dp1(v2)
        v4 = self.conv3(v3)
        v5 = torch.sigmoid(v4)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
