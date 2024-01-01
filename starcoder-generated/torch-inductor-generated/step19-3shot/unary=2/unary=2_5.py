
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), padding=(0, 0), output_padding=(0, 0))
        self.conv_t2 = torch.nn.ConvTranspose2d(32, 128, kernel_size=(3, 3), stride=(1, 1), dilation=(2, 2), padding=(2, 2), output_padding=1)
        self.conv_t3 = torch.nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(1, 1), dilation=(1, 1), padding=(0, 0), output_padding=0)
        self.conv_transpose = torch.nn.ConvTranspose2d(64, 16, kernel_size=(2, 2), stride=(1, 1), dilation=(2, 2), padding=(2, 2))
    def forward(self, x1):
        v1 = self.conv_t1(x1)
        v2 = self.conv_t2(v1)
        v3 = self.conv_t3(v2)
        v4 = self.conv_transpose(v3)
        v5 = v4 * 0.5
        v6 = v4 * v4 * v4
        v7 = v6 * 0.044715
        v8 = v4 + v7
        v9 = v8 * 0.7978845608028654
        v10 = torch.tanh(v9)
        v11 = v10 + 1
        v12 = v5 * v11
        return v12
# Inputs to the model
x1 = torch.randn(1, 64, 32, 32)
