
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(32, 64, 3, stride=2, padding=0, bias=False)
        self.group_norm = torch.nn.GroupNorm(8, 64)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return self.group_norm(v5)
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
