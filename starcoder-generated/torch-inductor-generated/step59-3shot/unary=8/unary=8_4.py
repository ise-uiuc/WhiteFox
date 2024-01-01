
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3, stride=1, padding=1)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(5, 9, 3, stride=2, padding=1, dilation=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose1(v1)
        v3 = v2 + 3
        v4 = torch.clamp(v3, min=0)
        v5 = torch.add(v4, 3, alpha=1)
        v6 = v5 + 3
        v7 = torch.clamp(v6, min=0)
        v8 = v3 * v7
        v9 = v8 / 6
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
