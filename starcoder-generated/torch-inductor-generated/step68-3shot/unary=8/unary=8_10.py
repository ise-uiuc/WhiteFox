
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, (7, 7), stride=(2, 2))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 30
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=30)
        v5 = v1 * v4
        v6 = v5 / 48
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 24, 24)
