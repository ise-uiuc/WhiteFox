
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 4, 4, stride=1, padding=1, bias=True)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 0.5
        v3 = torch.clamp(v2, min=None, max=None)
        v4 = v1 * v3
        v5 = v4 / 3
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
