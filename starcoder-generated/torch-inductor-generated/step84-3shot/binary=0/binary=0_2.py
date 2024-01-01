
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 63, 41, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(63, 73, 13, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(73, 3, 79, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 93, 96, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(93, 8, 47, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(8, 20, 93, stride=1, padding=1)
    def forward(self, x1, x2, x3, other=None):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2 + torch.randn(v2.shape).to(x2.device))
        v4 = self.conv4(v3 + other)
        v5 = self.conv5(v4)
        v6 = self.conv6(v5)
        v7 = v6 + x3
        return v7
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64).to('cpu')
x2 = torch.randn(1, 8, 64, 64).to('cpu')
x3 = torch.randn(1, 20, 64, 64).to('cpu')
