
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_wise_conv = torch.nn.ConvTranspose2d(32, 32, 3, stride=1, groups=32, padding=1)
    def forward(self, x1):
        v1 = self.depth_wise_conv(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 32, 512, 512)
