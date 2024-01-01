
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 4, 3, stride=1, padding=2, groups=2)
        self.conv3 = torch.nn.Conv2d(4, 4, 1, stride=1, padding=5)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1.add(10))
        v3 = v2.sub(2)
        v4 = self.conv3(v3 / 2)
        return v4.abs()
# Inputs to the model
x1 = torch.randn(3, 3, 7, 7)
