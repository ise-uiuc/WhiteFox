
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 1.73 + v1
        v3 = v2 / -3.01
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
