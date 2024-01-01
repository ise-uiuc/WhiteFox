
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 5, 3, stride=5, padding=2)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 > 1
        v3 = v1 * 2
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 5, 30, 20)
