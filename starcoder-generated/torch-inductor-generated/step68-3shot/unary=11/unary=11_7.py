
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, 3, stride=2, padding=0, groups=2)
    def forward(self, x1):
        x1 = x1.repeat(5, 1, 1, 1) # Repeat input tensor multiple times using the same weights
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 25, 25)
