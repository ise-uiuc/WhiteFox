
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(1, 9, 1, padding=0, bias=True)
    def forward(self, x1):
        v1 = self.conv_transpose(x1) + 3
        v2 = torch.clamp_min(v1, 0)
        v3 = torch.clamp_max(v2, 6)
        v4 = v3 / 6
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 16)
