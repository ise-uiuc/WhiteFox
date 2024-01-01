
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 64, 3, stride=1, padding=0, dilation=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(64, 64, 3, stride=1, padding=0, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        v3 = v1 + 3
        v4 = torch.clamp(v3, min=0)
        v5 = torch.clamp(v4, max=6)
        v6 = v1 * v5
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 20, 20)
