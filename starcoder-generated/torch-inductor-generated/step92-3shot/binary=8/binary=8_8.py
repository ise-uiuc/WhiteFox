
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1, stride=1)
        self.conv3 = torch.nn.Conv2d(3, 8, 3, padding=1, stride=1)
        self.conv4 = torch.nn.Conv2d(3, 8, 3, padding=1, stride=1)
        self.linear1 = torch.nn.Linear(10, 8)
        self.bn1 = torch.nn.BatchNorm2d(8)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v3 = self.conv3(x3)
        v4 = self.conv4(x2)
        v5 = v3 + v4
        v6 = v1 + v5
        return v6
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
x2 = torch.randn(2, 3, 32, 32)
x3 = torch.randn(2, 3, 32, 32)
