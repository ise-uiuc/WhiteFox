
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d1 = torch.nn.Conv2d(6, 6, 3, stride=2)
        self.conv2d2 = torch.nn.Conv2d(8, 6, 1)
        self.conv2d3 = torch.nn.Conv3d(in_channels=2, out_channels=4, kernel_size=(1, 1, 1), bias=True)
        self.conv3d1 = torch.nn.Conv3d(in_channels=2, out_channels=6, kernel_size=(1, 2, 3), stride=(2, 2, 2), groups=1, bias=False)
        self.conv3d2 = torch.nn.Conv3d(in_channels=2, out_channels=6, kernel_size=(3, 1, 1), padding=(2, 0, 0), bias=False)
        self.conv3d3 = torch.nn.Conv3d(in_channels=6, out_channels=6, kernel_size=(1, 1, 2), dilation=(2, 3, 2), padding=(1,0,1))
        self.bn2d = torch.nn.BatchNorm2d(8)
        self.bn3d = torch.nn.BatchNorm3d(6)
    def forward(self, x1):
        v1 = self.conv2d1(x1)
        v2 = self.conv2d2(self.bn2d(v1))
        v3 = self.conv2d3(self.bn2d(v1)) + v2
        v4 = self.conv3d1(v3)
        v5 = self.conv3d2(v2) + v4
        v6 = self.conv3d3(v5)
        v7 = self.bn3d(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 6, 9, 8)
