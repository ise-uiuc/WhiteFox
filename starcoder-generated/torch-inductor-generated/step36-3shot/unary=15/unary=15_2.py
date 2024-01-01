
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 1)
    def forward(self, x):
        v0, v1 = torch.chunk(x, 2, 1)
        v2 = self.conv1(v1)
        v3 = self.conv2(x)
        return torch.cat([v2, v3], dim=1)
# Inputs to the model
x = torch.randn(1, 64, 256, 256)
