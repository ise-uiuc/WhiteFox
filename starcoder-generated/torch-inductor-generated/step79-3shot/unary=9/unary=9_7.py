
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 10, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        out = nn.ReLU6(v2)
        return out
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
