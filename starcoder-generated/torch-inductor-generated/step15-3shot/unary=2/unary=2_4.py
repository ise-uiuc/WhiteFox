
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 24, kernel_size=(1, 5), stride=(1, 5), padding=6)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(24, 29, kernel_size=(3, 5), stride=(3, 5), padding=6)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(29, 42, kernel_size=(4, 5), stride=(4, 5), padding=6)
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
x1 = torch.randn(1, 4, 8, 232)
