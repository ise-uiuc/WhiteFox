
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
    def forward(self, x1, x2):
        m1 = x1 + x2
        m2 = x1 * x2
        m3 = x1 - x2
        m4 = x1 / (0.1 + x2)
        m5 = x1 + x2
        v1 = torch.cat([m2, m3, m4], dim=1)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v3)
        v5 = self.conv4(v3)
        v6 = self.conv5(v3)
        v7 = v5
        v7 = v5 + m5
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
x2 = torch.randn(1, 3, 16, 16)
