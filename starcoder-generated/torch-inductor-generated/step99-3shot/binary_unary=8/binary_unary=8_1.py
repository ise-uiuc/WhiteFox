
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 192, 1, bias=True, padding=0, stride=1, dilation=1, groups=1)
        self.conv2 = torch.nn.Conv2d(32, 192, 1, bias=True, padding=0, stride=1, dilation=1, groups=1)
        self.conv3 = torch.nn.Conv2d(32, 192, 1, bias=True, padding=0, stride=1, dilation=1, groups=1)
        self.conv4 = torch.nn.Conv2d(32, 192, 1, bias=True, padding=0, stride=1, dilation=1, groups=1)
        self.conv5 = torch.nn.Conv2d(32, 192, 1, bias=True, padding=0, stride=1, dilation=1, groups=1)
        self.conv6 = torch.nn.Conv2d(32, 192, 1, bias=True, padding=0, stride=1, dilation=1, groups=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = self.conv3(x)
        v4 = self.conv4(x)
        v5 = self.conv5(x)
        v6 = self.conv6(x)
        v7 = v1 + v2 + v3 + v4 + v5 + v6
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x = torch.randn(1, 32, 16, 16)
