
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 1, stride=2, padding=2)
        self.conv_next = torch.nn.Conv2d(64, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_next(v1)
        v3 = F.sigmoid(v2)
        v2 = v2.mul(v3)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
