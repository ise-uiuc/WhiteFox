
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, stride=1, padding=3)
        self.conv2d = torch.nn.Conv2d(3, 3, 1, stride=1, padding=2)
        self.conv3d = torch.nn.Conv3d(3, 3, 1, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv2d(x1)
        v3 = self.conv3d(x1)
        return v1 + v2 + v3
# Inputs to the model
x1 = torch.randn(1, 1, 4, 8)
