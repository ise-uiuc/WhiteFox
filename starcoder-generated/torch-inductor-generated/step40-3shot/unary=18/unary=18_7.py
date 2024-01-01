
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1)
        self.conv5 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1)
        self.conv6 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1)
        self.conv7 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1)
        self.conv8 = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = self.conv6(v5)
        v7 = self.conv7(v6)
        v8 = self.conv8(v7)
        v9 = torch.sigmoid(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
