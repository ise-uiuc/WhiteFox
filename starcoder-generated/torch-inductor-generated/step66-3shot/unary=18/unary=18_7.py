
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=38, out_channels=92, kernel_size=5, stride=3, padding=5)
        self.conv2 = torch.nn.Conv2d(in_channels=92, out_channels=88, kernel_size=1, stride=1, padding=1, groups=88)
        self.conv3 = torch.nn.Conv2d(in_channels=88, out_channels=24, kernel_size=1, stride=1, padding=1, groups=88)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v3)
        v5 = torch.sigmoid(v4)
        v6 = torch.sigmoid(v1)
        return v5
# Inputs to the model
x1 = torch.randn(1, 38, 22, 22)
