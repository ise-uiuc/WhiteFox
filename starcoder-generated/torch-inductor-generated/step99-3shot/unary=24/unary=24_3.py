
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 7, stride=1, padding=3, dilation=1, groups=1, bias=True)
        self.conv2 = torch.nn.Conv2d(1, 1, 7, stride=1, padding=3, dilation=2, groups=1, bias=True)
        self.conv3 = torch.nn.Conv2d(1, 1, 7, stride=1, padding=3, dilation=4, groups=1, bias=True)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        negative_slope = -0.0090389
        v1 = self.conv1(x) + 0.549
        v2 = self.conv2(v1) + 0.678
        v3 = self.conv3(v2) + 0.557
        v4 = self.relu(v3) * negative_slope
        v5 = self.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.cat((210 * torch.ones(1, 1, 101, 25), -56.3 * torch.ones(1, 1, 101, 25)))
