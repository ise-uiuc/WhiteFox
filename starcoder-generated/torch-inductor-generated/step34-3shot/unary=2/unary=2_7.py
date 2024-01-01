
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 4, stride=1, padding=2, dilation=2, groups=1, bias=True)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = self.sigmoid(v6)
        v8 = v7 + 1
        v2 = v1 * 0.5
        v9 = v2 * v8
        return v9
# Inputs to the model
x1 = torch.randn(1024, 3, 32, 32)
