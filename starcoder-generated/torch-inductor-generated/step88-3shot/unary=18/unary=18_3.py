
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1, padding=1, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, padding=1, stride=1)
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding=1, stride=1)
        self.conv5 = torch.nn.Conv2d(in_channels=16, out_channels=4, kernel_size=2, padding=1)
        self.conv6 = torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.conv7 = torch.nn.Conv2d(in_channels=8, out_channels=4, kernel_size=2, padding=1, stride=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = self.conv6(v5)
        v7 = self.conv7(v6)
        v8 = torch.sigmoid(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
