
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(7, 5, 2, stride=3, padding=1)
        self.conv_transpose = torch.nn.ConvTranspose3d(5, 8, 2, stride=3, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose(v1)
        v3 = v2 * 0.5
        v4 = v2 + 1
        v5 = v3 * v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 7, 128, 128, 18)
