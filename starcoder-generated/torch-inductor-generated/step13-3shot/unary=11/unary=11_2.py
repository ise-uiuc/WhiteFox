
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = torch.clamp_min(x1, 0)
        v2 = torch.clamp_max(v1, 6)
        v3 = self.conv_transpose(v2)
        v4 = v3 + 3
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
