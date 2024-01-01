
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(16, 120, 5, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.avg_pool2d(v1, 2, stride=2, padding=0, ceil_mode=False)
        v3 = self.conv2(v2)
        v4 = torch.flatten(v3, 1)
        v5 = self.conv3(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
