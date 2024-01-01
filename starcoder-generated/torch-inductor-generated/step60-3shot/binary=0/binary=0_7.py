
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(45, 8, 1, stride=1, padding=1)
    def forward(self, x1, y1, other=None):
        v1 = self.conv(x1)
        z = y1 * other
        v2 = v1 + z
        return v2
# Inputs to the model
x1 = torch.randn(1, 45, 64, 64).to('cpu')
y1 = torch.randn(1, 45, 64, 64).to('cpu')
