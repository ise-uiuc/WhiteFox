
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(12, 640, 9, stride=4, padding=0, bias=False)
        self.conv2 = torch.nn.Conv2d(640, 1280, 1, stride=1, padding=0, bias=False)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.sigmoid(v2)
        v4 = x2 / v3
        v5 = 1 + v4
        return v3
# Inputs to the model
x1 = torch.randn(1, 12, 64, 64)
x2 = torch.randn(1, 1280, 1, 1)
