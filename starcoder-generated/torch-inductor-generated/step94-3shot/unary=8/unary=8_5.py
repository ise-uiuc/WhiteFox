
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1, output_padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1.flatten(1, 2)
        v3 = v2 + 3
        v4 = torch.clamp(v3, min=0)
        v5 = torch.clamp(v4, max=6)
        v6 = v2 * v5
        v7 = v6.view(1, 1, 1, 4)
        v8 = v7 / 6
        return v8
# Inputs to the model
x1 = torch.randn(1, 16, 17, 17)
