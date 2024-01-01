
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.conv2(v1)
        v3 = v2.sum() - 8.1852
        return v3
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
