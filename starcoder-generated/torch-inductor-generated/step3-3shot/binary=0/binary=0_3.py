
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, b, size):
        v1 = self.conv(x1)
        v2 = v1 + b
        return v2[..., :size[0], :size[1]]
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
b = torch.randn(1, 1, 64, 64)
size = (16, 16)
