
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(6, 6, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 100
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 128)
        v5 = v4 / 64
        return v5
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
