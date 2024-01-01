
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(17, 17, 15, stride=1, padding=0, dilation=13, groups=17)
        self.conv2 = torch.nn.Conv2d(17, 32, 1, stride=1, padding=0, dilation=1, groups=17)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v1.sigmoid()
        v2 = self.conv2(x1)
        v2.sigmoid()
        return torch.cat((v1, v2), 0)
# Inputs to the model
x1 = torch.randn(1, 17, 64, 64)
