
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.conv2d(3, 8, 1, stride=1, padding=1)
        v2 = v1 + 3
        v3 = v2.clamp(min=0, max=6)
        v4 = v3 / 6
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
