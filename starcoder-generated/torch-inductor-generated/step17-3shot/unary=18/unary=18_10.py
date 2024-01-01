
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=2, stride=2, padding=0)
        self.conv5 = torch.nn.Conv2d(in_channels=32, out_channels=8, kernel_size=1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.conv7 = torch.nn.Conv2d(in_channels=64, out_channels=8, kernel_size=1, stride=1, padding=0)
        self.conv8 = torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.sigmoid(v2)
        v4 = self.conv3(v3)
        v5 = self.conv4(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv5(v6)
        v8 = self.conv6(v7)
        v9 = torch.sigmoid(v8)
        v10 = self.conv7(v9)
        v11 = self.conv8(v10)
        v12 = torch.sigmoid(v11)
        return v12
# Inputs to the model
x1 = torch.randn(1, 3, 2, 3)
