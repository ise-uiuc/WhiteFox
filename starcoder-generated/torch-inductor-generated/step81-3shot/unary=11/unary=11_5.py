
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 6, 2, stride=2, padding=0)
    def forward(self, x2):
        v1 = self.conv_transpose(x2)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 2)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 0
        return v5
# Inputs to the model
x2 = torch.randn(1, 4, 32, 32)
