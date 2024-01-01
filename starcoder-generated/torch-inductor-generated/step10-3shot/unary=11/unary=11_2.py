
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 10, stride=1, padding=1)
    def forward(self, x1):
        v1 = torch.clamp_min(x1, 1)
        v2 = torch.clamp_max(v1, 5)
        v3 = self.conv_transpose(v2)
        v4 = v3 + 3
        v5 = torch.clamp_min(v4, 0)
        v6 = torch.clamp_max(v5, 6)
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
