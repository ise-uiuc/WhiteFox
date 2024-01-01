
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 4, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = x1.permute(0, 1, 3, 2).contiguous()
        v2 = self.conv_transpose(v1)
        v3 = v2.permute(0, 1, 3, 2).contiguous()
        v4 = v3 + 3
        v5 = torch.clamp_min(v4, 0)
        v6 = torch.clamp_max(v5, 6)
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
