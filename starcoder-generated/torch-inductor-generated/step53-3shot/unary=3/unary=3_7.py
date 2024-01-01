
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(2, 1, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(1, 4, 3, stride=3, padding=5)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 23, 40)
