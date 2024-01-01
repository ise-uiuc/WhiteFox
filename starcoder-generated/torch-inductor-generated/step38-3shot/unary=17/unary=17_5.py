
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(1, 128, (1, 16), stride=[1, 2])
        self.bn1 = torch.nn.BatchNorm2d(num_features=128, eps=2e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.ConvTranspose3d(128, 128, (4, 16, 16), stride=1, padding=1, output_padding=0, bias=False, dilation=1, groups=1)
        self.conv3 = torch.nn.ConvTranspose3d(128, 64, (4, 2, 2), stride=[2, 1, 1], padding=0, output_padding=0, bias=False, dilation=1, groups=1)
        self.conv4 = torch.nn.ConvTranspose2d(64, 64, (2, 2), stride=[1, 1], padding=0, output_padding=0, bias=False, dilation=1, groups=1)
        self.conv5 = torch.nn.ConvTranspose2d(64, 32, (2, 2), stride=[1, 1], padding=0, output_padding=0, bias=False, dilation=1, groups=1)
        self.conv6 = torch.nn.ConvTranspose3d(32, 32, (4, 4, 4), stride=[1, 1, 1], padding=0, output_padding=0, bias=False, dilation=1, groups=1)
        self.conv7 = torch.nn.Conv2d(32, 4, (2, 2), stride=[1, 1], padding=0, output_padding=0, bias=True, dilation=1, groups=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = self.relu1(v2)
        v4 = self.conv2(v3)
        v5 = self.conv3(v4)
        v6 = self.conv4(v5)
        v7 = self.conv5(v6)
        v8 = self.conv6(v7)
        v9 = self.conv7(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
