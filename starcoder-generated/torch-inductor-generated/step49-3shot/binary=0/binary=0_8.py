
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 4, stride=2, padding=3)
        self.conv3 = torch.nn.Conv2d(8, 4, 5, stride=1, padding=10)
    def forward(self, x1, other=1, padding1=None):
        v1 = self.conv1(x1)
        if not padding1 is None:
            v1 += padding1
        v2 = self.conv2(v1) + other
        v3 = self.conv3(v2)
        v4 = v3 + other
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 640, 640)
