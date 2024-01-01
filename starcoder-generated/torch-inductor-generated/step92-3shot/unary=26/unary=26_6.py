
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(47, 559, 5, stride=2, padding=2, bias=False)
    def forward(self, x2):
        v1 = self.conv_t(x2)
        v2 = -3.63 + v1
        v3 = v2 > 0
        v4 = v2 * 1.1133
        v5 = torch.where(v3, v2, v4)
        v6 = -0.0541 + v1
        v7 = v6 > 0
        v8 = v6 * -2.0312
        v9 = torch.where(v7, v6, v8)
        v10 = v5 + v9
        return v10
# Inputs to the model
x2 = torch.randn(2, 47, 48, 39)
