
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1, groups=3)
        self.conv4 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = self.conv3(x)
        v4 = self.conv4(x)
        v5 = self.conv5(x)
        v6 = self.conv6(x)
        x = v1 + v2 + v3
        x = x + v4 + v5
        x = x + v6
        return x
# Inputs to the model
x = torch.randn(1, 3, 127, 127)
