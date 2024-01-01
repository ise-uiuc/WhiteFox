
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv2(x2)
        v4 = self.conv1(x2)
        v5 = self.conv4(x1)
        v6 = v5 + v1
        v7 = self.conv3(x2)
        v8 = v7 + v4
        v9 = v2 + v6
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
