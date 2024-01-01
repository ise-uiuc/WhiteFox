
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1)
        self.conv3 = torch.nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 16, 1, stride=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = v4 - 1.5
        v6 = v5 + 4
        return v6
# Inputs to the model
x = torch.randn(1, 1, 64, 64)
