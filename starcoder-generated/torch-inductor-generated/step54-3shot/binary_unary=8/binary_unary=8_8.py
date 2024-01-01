
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 3, 1, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(64, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv3(x1)
        v2 = self.conv4(v1)
        v3 = self.conv5(x1)
        v4 = self.conv6(v3)
        v5 = v2 + v4
        return v5
# Inputs to the model
x1 = torch.randn(2, 3, 32, 32)
