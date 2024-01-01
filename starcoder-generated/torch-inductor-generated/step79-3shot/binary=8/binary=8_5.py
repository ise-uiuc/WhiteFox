
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1, dilation=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1, dilation=1)
        self.conv3 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=0, dilation=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(x)
        v4 = v2 + v3
        s1 = v4.sum()
        return s1
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
