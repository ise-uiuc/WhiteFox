
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=4, dilation=3)
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=8, dilation=4)
        self.conv5 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=12, dilation=5)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv4(v6)
        v8 = torch.sigmoid(v7)
        v9 = self.conv5(v8)
        v10 = torch.sigmoid(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 64, 112, 112)
