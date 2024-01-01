
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 64, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.rand_like(v1) + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 3)
        v5 = torch.rand_like(v1) + 0.3
        v6 = torch.clamp_min(v5, 0)
        x2 = torch.abs(v4 / v6)
        return x2
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
