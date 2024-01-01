
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 16, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(8, 16, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = torch.flatten(v1)
        v4 = torch.flatten(v2)
        v5 = v3 + v4
        v6 = self.conv3(v5)
        v7 = self.conv4(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 8, 256, 256)
x2 = torch.randn(1, 8, 256, 256)
