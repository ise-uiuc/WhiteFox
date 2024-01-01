
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 6, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(6, 3, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 6, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = v2 + v1
        v4 = torch.clamp(v3, 0, 6)
        v5 = self.conv3(v4)
        v6 = v4 * v5
        v7 = v6 / 6
        v8 = self.conv4(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
