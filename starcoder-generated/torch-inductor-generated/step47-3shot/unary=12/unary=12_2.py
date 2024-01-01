
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = nn.Conv2d(3, 6, 1, stride=2, padding=1, dilation=1, groups=1)(x1)
        v2 = torch.sigmoid(v1)
        v3 = torch.mul(v1, v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
