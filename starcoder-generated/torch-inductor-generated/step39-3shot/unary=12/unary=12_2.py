
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=2, dilation=2)
        self.conv3 = torch.nn.Conv2d(8, 16, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(16, 32, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(v1)
        v4 = self.conv4(v2)
        v5 = self.conv5(v3)
        v6 = torch.cat((v3, v4, v5), dim=1)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
