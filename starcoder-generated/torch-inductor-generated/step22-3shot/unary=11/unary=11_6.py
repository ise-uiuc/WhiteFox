
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, 0) + 3
        v3 = torch.clamp_max(v2, 6)
        v4 = v3 / 6
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
