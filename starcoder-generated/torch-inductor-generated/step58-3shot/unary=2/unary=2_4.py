
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(32, 4, kernel_size=(3, 5), stride=(2, 1), padding=(1, 1), output_padding=(4, 2), groups=1, dilation=(2, 1), bias=False)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        return v9
# Inputs to the model
x1 = torch.ones(1, 32, 17, 14)
