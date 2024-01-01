
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 1, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(1, 128, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 1, 3, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(1, 256, 3, stride=2, padding=1)
        self.conv6 = torch.nn.Conv2d(256, 1, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v1_1 = self.conv2(v1)
        v2 = self.conv3(x1)
        v2_1 = self.conv4(v2)
        v3 = self.conv5(x1)
        v3_1 = self.conv6(v3)
        v4 = torch.cat([v1_1, v2_1, v3_1], 1)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 51, 62)
