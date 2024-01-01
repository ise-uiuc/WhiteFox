
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        x5 = torch.randn(1, 32, 64, 64)
        v5 = self.conv1(x5)
        v6 = torch.sigmoid(v5)
        x7 = torch.randn(1, 64, 64, 64)
        v7 = torch.sigmoid(v7)
        x8 = torch.cat((x2, v5, v7), 1)
        v8 = self.conv2(x8)
        v9 = torch.sigmoid(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
