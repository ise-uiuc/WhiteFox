
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(2, 3, 1, stride=1, padding=1, dilation=2, groups=3, output_padding=1, output_size=(28, 20))
        self.conv_transpose2 = torch.nn.ConvTranspose2d(3, 1, 2, stride=2, padding=1, dilation=2, groups=3, output_padding=1, output_size=(70, 75))
    def forward(self, x1, x2):
        v1 = self.conv_transpose1(torch.cat((x1, x2), dim=1)).to(torch.float64)
        v2 = self.conv_transpose2(v1).to(torch.float64)
        v3 = v2 + 3
        v4 = torch.clamp(v3, min=0)
        v5 = torch.clamp(v4, max=6)
        v6 = v2 * v5
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 26, 50)
x2 = torch.randn(1, 4, 24, 24)
