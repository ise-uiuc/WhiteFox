
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depthwise_conv_transpose = torch.nn.ConvTranspose2d(32, 32, 10, groups=32, stride=2, padding=5)
    def forward(self, x1):
        v1 = self.depthwise_conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 4, 8, 8)
