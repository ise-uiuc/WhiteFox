
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 48, 5, stride=1, padding=(2, 2), output_padding=(2, 2))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 16, 32, 32)
