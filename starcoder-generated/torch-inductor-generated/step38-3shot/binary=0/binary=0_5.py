
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=2, padding=0)
    def forward(self, x1, x2, other):
        v1 = self.conv(x1 + other)
        v2 = v1 + other.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        v3 = v2 + other
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 32, 32)
other = torch.randn(1, 3, 32, 32)
# model ends