
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 1, 3, stride=2, padding=1, output_padding=(0,1))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 35, 35)
x2 = torch.randn(1, 1, 35, 35)
x3 = torch.randn(1, 2, 18, 18)
x4 = torch.randn(1, 2, 18, 18)
x5 = torch.randn(1, 2, 28, 28)
x6 = torch.randn(2, 2, 28, 28)
