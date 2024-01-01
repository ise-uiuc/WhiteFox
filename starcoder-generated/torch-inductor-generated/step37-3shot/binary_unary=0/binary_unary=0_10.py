
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 32, 2, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = x * v1
        v3 = x + v2
        return v3
# Inputs to the model
x = torch.randn(1, 32, 32, 32)
