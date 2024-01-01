
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(84, 32, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + v1
        v3 = v2 + v2
        v4 = torch.sigmoid(v3)
        v5 = v3 * v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 84, 112, 112)
