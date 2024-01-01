
class Model(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 5, stride=1, padding=2)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * 0.1
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
in_dim = 32
out_dim = 64
x1 = torch.randn(1, 3, 64, 64)
