
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(512, 1024, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 < -0.02
        v3 = v1 - 0.12 * v2
        return v3
# Inputs to the model
x1 = torch.randn(2, 512, 8, 8)
