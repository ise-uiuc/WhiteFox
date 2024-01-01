
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(7, 31, 2, stride=(1, 1), padding=(1, 1), output_padding=0, groups=1, bias=False)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(31, 16, 2, stride=(1, 1), padding=(3, 0), output_padding=(0, 0), groups=1, bias=False)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(16, 52, 2, stride=(1, 1), padding=(1, 2), output_padding=(0, 1), groups=1, bias=False)
        self.conv_transpose4 = torch.nn.ConvTranspose2d(52, 13, 2, stride=(1, 1), padding=(1, 1), output_padding=0, groups=1, bias=False)
        self.conv_transpose5 = torch.nn.ConvTranspose2d(13, 47, 2, stride=(1, 1), padding=(0, 2), output_padding=(0, 0), groups=1, bias=False)
        self.conv_transpose6 = torch.nn.ConvTranspose2d(47, 28, 2, stride=(1, 1), padding=(0, 0), output_padding=(0, 0), groups=1, bias=False)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v1 = self.conv_transpose2(v1)
        v1 = self.conv_transpose3(v1)
        v1 = self.conv_transpose4(v1)
        v1 = self.conv_transpose5(v1)
        v1 = self.conv_transpose6(v1)
        v1 = self.conv_transpose3(v1)
        v1 = self.conv_transpose4(v1)
        v1 = self.conv_transpose5(v1)
        v1 = self.conv_transpose6(v1)
        v1 = self.conv_transpose2(v1)
        v1 = self.conv_transpose3(v1)
        v1 = self.conv_transpose4(v1)
        v1 = self.conv_transpose5(v1)
        v1 = self.conv_transpose6(v1)
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
x1 = torch.randn(1, 7, 2, 2)
