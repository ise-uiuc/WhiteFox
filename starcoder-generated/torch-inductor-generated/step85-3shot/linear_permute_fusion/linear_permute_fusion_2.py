
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True)
        self.relu = torch.nn.ReLU(inplace=True)
        self.flatten = torch.nn.Flatten()
    def forward(self, x):
        v1 = self.conv1(x)
        v1 = self.relu(v1)
        v2 = self.conv2(v1)
        v2 = torch.transpose(v2, 1, 3)
        v3 = self.relu(v2)
        v4 = v3.size(3)
        v5 = v3.size(2)
        v6 = v4 * v5
        v6 = v6.int()
        v6 = v6.__floordiv__(4)
        v7 = v6.size(0)
        v8 = -1 if v7 <= 0 else v7 - 4
        v9 = 1
        v10 = self.flatten(v3[..., v8:v9])
        return v10
# Inputs to the model
x = torch.randn(1, 1, 7, 7)
