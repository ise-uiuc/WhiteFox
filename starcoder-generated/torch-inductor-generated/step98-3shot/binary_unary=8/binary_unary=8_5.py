
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, (1, 3), stride=(1, 1), padding=(0, 1))
        self.conv2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv3 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv4 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(x1)
        v4 = v1 + v2 + v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
