
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(x2)
        v3 = self.conv2(x1)
        a1 = self.conv2(v2) + self.conv4(torch.sigmoid(v1)) + v3
        v4 = torch.sigmoid(a1)
        v5 = self.conv2(x1) + self.conv3(torch.sigmoid(v4))
        return v5
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
