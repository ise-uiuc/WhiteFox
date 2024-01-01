
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(24, 7, stride=1, kernel_size=1, padding=0, output_padding=0, groups=1, bias=False)
        self.pad = torch.nn.ConstantPad2d(padding=(0, 0, 0, 0), value=0.)
    def forward(self, x1):
        x = x1
        x4 = self.conv_transpose(x)
        x6 = self.pad(x4)
        v1 = x6 * 0.5
        v2 = x6 * x6 * x6
        v3 = v2 * 0.044715
        v4 = x6 + v3
        v5 = v4 * 0.7978845608028654
        v6 = torch.tanh(v5)
        v7 = v6 + 1
        v8 = v1 * v7
        return v8
# Inputs to the model
x1 = torch.randn(1, 24, 7, 9)
