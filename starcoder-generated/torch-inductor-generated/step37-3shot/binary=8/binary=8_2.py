
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1, dilation=2)
        self.conv4 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=3, dilation=2)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.conv3(x3)
        v4 = self.conv4(x4)
        v3 = v3.sum(1,keepdim=True).sum(2,keepdim=True).sum(3,keepdim=True)
        v4 = v4.sum(2,keepdim=True).sum(3,keepdim=True).sum(4,keepdim=True)
        v3 += v4
        v5 = v1 + v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
x4 = torch.randn(1, 3, 64, 64)
