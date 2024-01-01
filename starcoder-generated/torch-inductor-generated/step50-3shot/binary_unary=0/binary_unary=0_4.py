
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(2, 3, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(3, 256, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2
        v4 = self.conv3(v3)
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 256, 14, 14)
x2 = torch.randn(1, 2, 64, 64)
x3 = torch.randn(1, 256, 14, 14)
