
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=8, out_channels=64, kernel_size=7, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.batch_norm = torch.nn.BatchNorm2d(num_features=64)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = torch.sigmoid(v4)
        v6 = self.batch_norm(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
