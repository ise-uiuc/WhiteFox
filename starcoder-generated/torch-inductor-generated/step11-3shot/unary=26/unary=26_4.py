
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(3, 3, 3)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 > 0
        v3 = v1 * 0.67
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(6, 3, 10)
