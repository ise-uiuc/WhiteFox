
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dconv_transpose = torch.nn.ConvTranspose2d(16, 4, 3, stride=1, padding=0, dilation=2)
    def forward(self, x1):
        v1 = self.dconv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 16, 32, 32)
