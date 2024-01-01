
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv1(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv2(v6)
        v8 = torch.sigmoid(v7)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
