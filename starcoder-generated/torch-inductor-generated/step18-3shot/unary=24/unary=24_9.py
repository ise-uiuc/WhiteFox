
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
    def forward(self, x):
        padding = 1
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * 0.1
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 17, 1)
