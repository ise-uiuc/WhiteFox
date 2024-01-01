
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 14, kernel_size=2, stride=2, padding=1, dilation=1, groups=1, bias=True)
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv1 = torch.nn.Conv2d(14, 12, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.add = torch.add
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.relu(v1)
        v3 = self.conv1(v2)
        v4 = self.add(v1, v3)
        v5 = v4 - 1.6
        return v5
# Inputs to the model
x = torch.randn(1, 3, 10, 12)
