
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 10, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = self.conv(x2)
        v3 = torch.sigmoid(v1 + v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 10, 64, 64)
x2 = torch.randn(1, 10, 64, 64)
