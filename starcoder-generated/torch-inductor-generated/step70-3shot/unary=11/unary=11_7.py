
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2d_1 = torch.nn.ConvTranspose2d(1, 100, 7, stride=2, padding=3)
        self.conv_transpose2d_2 = torch.nn.ConvTranspose2d(100, 100, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose2d_1(x1)
        v2 = self.conv_transpose2d_2(v1)
        v3 = v2 + 3
        v4 = torch.clamp_min(v3, 0)
        v5 = torch.clamp_max(v4, 6)
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
