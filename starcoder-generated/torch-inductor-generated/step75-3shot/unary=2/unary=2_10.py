
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose3d(8, 2, kernel_size=(2, 1, 4), stride=(2, 1, 1), padding=(0, 0, 2), output_padding=(0, 0, 0))
        self.conv_transpose2 = torch.nn.ConvTranspose3d(2, 4, kernel_size=(1, 3, 3), stride=(1, 1, 2), padding=(0, 1, 2))
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        v3 = v2 * 0.5
        v4 = v2 * v2 * v2
        v5 = v4 * 0.044715
        v6 = v2 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v3 * v9
        return v10
# Inputs to the model
x1 = torch.randn(2, 8, 2, 7, 12)
