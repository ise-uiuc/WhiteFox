
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 11, 2, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(11, 12, 2, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(12, 13, 2, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(13, 14, 2, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v1 = self.conv2(v1)
        v1 = self.conv3(v1)
        v1 = self.conv4(v1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(2, 3, 28, 28)
