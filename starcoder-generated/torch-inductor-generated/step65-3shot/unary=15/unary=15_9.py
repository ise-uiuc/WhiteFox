
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.bias_conv1 = torch.nn.Parameter(torch.randn(64))
        self.bn1 = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.010000000000000009, affine=True)
        self.conv2 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.bias_conv2 = torch.nn.Parameter(torch.randn(64))
        self.bn2 = torch.nn.BatchNorm2d(64, eps=0.001, momentum=0.010000000000000009, affine=True)
        self.conv3 = torch.nn.Conv2d(64, 64, 1, stride=2, bias=False)
        self.conv4 = torch.nn.Conv2d(64, 128, 1, stride=1, bias=False)
        self.conv5 = torch.nn.Conv2d(128, 128, 1, stride=1, bias=False)
        self.conv6 = torch.nn.Conv2d(128, 128, 1, stride=1, bias=False)
        self.conv7 = torch.nn.Conv2d(128, 128, 1, stride=1, bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.add(v1, self.bias_conv1)
        v3 = self.bn1(v2)
        v4 = self.conv2(v3)
        v5 = torch.add(v4, self.bias_conv2)
        v6 = self.bn2(v5)
        v7 = self.conv3(v6)
        v8 = self.conv4(v7)
        v9 = self.conv5(v6)
        v10 = self.conv6(v7)
        v11 = self.conv7(v8)
        return v11
# Inputs to the model
x1 = torch.randn(1, 64, 512, 1024)
