
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        t1 = torch.nn.ConvTranspose2d(32, 1, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
        t2 = torch.nn.ReLU()

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
x1 = torch.randn(1, 32, 7, 7)
