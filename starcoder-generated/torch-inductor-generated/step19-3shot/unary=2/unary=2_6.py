
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 8, 4, stride=(2, 2), output_padding=(1, 1), padding=(2, 2), groups=2)
        self.max = torch.nn.ConvTranspose2d(8, 6, (3, 3), stride=(2, 2), dilation=(1, 1), padding=(1, 1), output_padding=(1, 1))
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = self.max(v1)
        v3 = v2 * 0.5
        v4 = v2 * v2 * v2
        v5 = v4 * 0.044715
        v6 = v2 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v3 * v8
        return v10
# Inputs to the model
x1 = torch.randn(1, 4, 4, 4)
