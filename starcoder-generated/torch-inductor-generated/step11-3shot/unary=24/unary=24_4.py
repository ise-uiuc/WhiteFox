
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 > 0
        v4 = torch.where(v2, v1 * -0.6, v1)
        return v4
# Inputs to the model
x1 = torch.randn(1, 4, 128, 128)
