
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 16, 3, stride=3, padding=2, dilation=2)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 - 5
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 * 5
        return v5
# Inputs to the model
x1 = torch.randn(1, 4, 60, 60)
