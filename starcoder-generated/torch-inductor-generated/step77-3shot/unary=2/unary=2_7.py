
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv_3d = torch.nn.ConvTranspose3d(1, 1, 2, stride=2, output_padding=1)
    def forward(self, x2):
        x = x2
        x4 = self.deconv_3d(x)
        v2 = x4 * 0.5
        v3 = x4 * x4 * x4
        v4 = v3 * 0.044715
        v5 = x4 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        return v9
# Inputs to the model
x2 = torch.randn(1, 1, 4, 16, 5)
