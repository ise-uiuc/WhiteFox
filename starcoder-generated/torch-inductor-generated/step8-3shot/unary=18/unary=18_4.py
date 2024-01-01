
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(in_channels=3, out_channels=128, kernel_size=1, stride=2, padding=0)
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(x1)
        v4 = torch.sigmoid(v3)
        v5 = self.conv4(v4)
        v6 = self.conv5(v5)
        v7 = torch.sigmoid(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 32, 64)
