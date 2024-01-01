
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 32, 3, stride=2)
        self.conv_transpose = torch.nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.conv_transpose(v1)
        v3 = v2 + 3
        v4 = torch.clamp_min(v3, 0)
        v5 = torch.clamp_max(v4, 6)
        v6 = v5 / 6
        return v1 + v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
