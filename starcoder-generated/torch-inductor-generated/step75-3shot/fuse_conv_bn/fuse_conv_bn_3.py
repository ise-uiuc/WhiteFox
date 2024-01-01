
class M1(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4):
        v1 = F.interpolate(x1, None, 2, 3, True, 'zeros', self.training)
        v1 = v1 + x2
        v1 = F.instance_norm(v1)
        v1 = v1 + x3
        v1 = v1.expand(1, -1, 256, 1860)
        v1 = v1 * x4
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 128, 1860)
x2 = torch.randn(1, 3, 256, 1860)
x3 = torch.randn(1, 1, 256, 1860)
x4 = torch.randn(1, 1860)
