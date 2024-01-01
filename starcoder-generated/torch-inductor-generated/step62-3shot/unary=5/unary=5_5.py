
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 8, 2, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 0.5
        v3 = v1 + 0.125
        v4 = v1 + 0.0625
        v5 = v1 + 0.03125
        v6 = v1 + 0.01562
        v7 = v1 + 0.0078125
        return v7
# Inputs to the model
x1 = torch.randn(1, 8, 32, 32)
