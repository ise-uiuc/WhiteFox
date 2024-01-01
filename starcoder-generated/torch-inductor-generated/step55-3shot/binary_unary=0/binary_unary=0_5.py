
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(17, 19, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(19, 17, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(17, 18, 3, stride=1, padding=1)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v1 + x2
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        v6 = v5 + v4
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 17, 64, 64)
x2 = torch.randn(1, 17, 64, 64)
x3 = torch.randn(1, 17, 64, 64)
