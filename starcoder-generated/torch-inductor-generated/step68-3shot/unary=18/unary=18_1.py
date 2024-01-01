
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=23, stride=21)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=9, kernel_size=5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels=9, out_channels=12, kernel_size=3)
        self.conv4 = torch.nn.Conv2d(in_channels=12, out_channels=15, kernel_size=7, stride=11, padding=3, dilation=2)
    def forward(self, x1):
        # TODO(qinjian2020): Your code here.
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv4(v6)
        v8 = torch.sigmoid(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
