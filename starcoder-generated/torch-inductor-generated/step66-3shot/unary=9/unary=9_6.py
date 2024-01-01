
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, padding=2)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, padding=1)
    def forward(self, x1):
        v2 = 3 + self.conv1(x1)
        v3 = v2.clamp(0, 6)
        v4 = v3 / 6
        return self.conv2(v4)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
