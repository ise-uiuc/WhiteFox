
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(11, 11, 1, stride=1, padding=5)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * 3.146
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 11, 30, 20)
