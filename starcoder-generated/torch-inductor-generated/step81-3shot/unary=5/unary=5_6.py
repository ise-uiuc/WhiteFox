
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(1, 8, 3, stride=2, bias=True, padding=2)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv_transpose(x1)
        v3 = v1 * 0.7322775509932292
        v4 = torch.log(v1)
        v5 = torch.erfinv(v2)
        v6 = torch.erf(v3)
        v7 = v6 + 1
        v8 = v5 * v7
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 3, 3, 3)
