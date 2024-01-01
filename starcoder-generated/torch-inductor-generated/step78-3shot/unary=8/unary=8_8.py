
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 3, 2)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 3
        v3 = v2 + 6
        v4 = torch.clamp(v3, min=0)
        v5 = torch.clamp(v4, max=30)
        v6 = v3 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 15, 15)
