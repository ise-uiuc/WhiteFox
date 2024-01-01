
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 8, 6, stride=5, dilation=5, padding=15)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.4431026511153168
        v3 = v1 * 0.6858517554209535
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 4, 45, 45)
