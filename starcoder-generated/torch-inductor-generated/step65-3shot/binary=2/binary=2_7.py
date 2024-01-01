
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 12, 5, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(12, 18, 5, stride=3, padding=3)
        self.conv3 = torch.nn.Conv2d(18, 24, 5, stride=3, padding=3)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3 - 5.0
        return v4
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
