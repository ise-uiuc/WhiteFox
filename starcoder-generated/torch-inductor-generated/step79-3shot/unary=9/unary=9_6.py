
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)
    def forward(self, x1):
        v1 = self.conv(x1)
        x2 = 3 + v1
        x3 = torch.clamp(x2, 0, 6)
        x4 = torch.sigmoid(x3 / 6 * 2)
        return x4
# Input to the model
x1 = torch.randn(1, 3, 224, 224)
