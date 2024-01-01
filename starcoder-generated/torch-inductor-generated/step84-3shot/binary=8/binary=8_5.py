
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 12, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(3, 20, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.conv3(x1)
        v4 = self.conv4(x2)
        v5 = self.conv5(x1)
        v6 = v1 + v2
        v7 = v3 + v4
        v8 = v5 + v6
        v9 = v7 + v8
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
