
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose0 = torch.nn.ConvTranspose2d(128, 16, 3, stride=1, padding=1, output_padding=0)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(50, 8, 3, stride=1, padding=1, output_padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose0(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6

        v6 = self.conv_transpose1(v5)
        v7 = v6 + 3
        v8 = torch.clamp_min(v7, 0)
        v9 = torch.clamp_max(v8, 6)
        v10 = v9 / 6

        return v10
# Inputs to the model
x1 = torch.randn(1, 128, 64, 64)
x2 = torch.randn(1, 50, 40, 40)
