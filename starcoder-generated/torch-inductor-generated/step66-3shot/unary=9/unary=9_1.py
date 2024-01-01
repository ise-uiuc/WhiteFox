
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1)
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=1)
        self.conv4 = torch.nn.Conv2d(3, 8, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(x1)
        v4 = self.conv4(x1)
        v6 = v1 + v2 + v3 + v4
        return v6
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
