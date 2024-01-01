
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(35, 22, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(22, 32, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 29, 1, stride=1, padding=1)
    def forward(self, x1, other):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1 + other)
        v3 = self.conv3(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 35, 64, 64)
other = torch.randn(1, 29, 64, 64)
