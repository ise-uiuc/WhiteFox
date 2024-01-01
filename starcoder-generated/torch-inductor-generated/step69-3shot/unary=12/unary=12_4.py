
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.conv1 = torch.nn.Conv2d(3, 18, 3, 2, 1)
        self.conv2 = torch.nn.Conv2d(16, 18, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(18, 18, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(18, 18, 3, 1, 1)
        self.conv5 = torch.nn.Conv2d(18, 3, 3, 1, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.sigmoid(v1)
        v3 = v1 * v2
        v4 = torch.cat([v3, x1], dim=1)
        v4 = self.conv2(v4)
        v4 = self.conv3(v4)
        v4 = self.conv4(v4)
        v5 = self.conv5(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
