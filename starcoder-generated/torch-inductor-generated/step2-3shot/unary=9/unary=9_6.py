
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v3 = self.conv2(self.conv(x1))
        v4 = v3 / 3
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
