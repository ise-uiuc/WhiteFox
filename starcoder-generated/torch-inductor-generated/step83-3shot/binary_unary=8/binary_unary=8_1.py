
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2d2 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv2d1(x1)
        v2 = self.conv2d2(x1)
        v3 = torch.relu(v2 + v1)
        v4 = self.conv2d1(v3)
        v5 = torch.relu(v4)
        v6 = self.conv2d2(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
