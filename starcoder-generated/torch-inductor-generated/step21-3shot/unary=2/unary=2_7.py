
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(49, 9, kernel_size=3, padding=1, stride=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(9, 42, kernel_size=3, padding=2, stride=2, output_padding=(1, 1))
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        v10 = self.conv_transpose2(v9)
        return v10
# Inputs to the model
x1 = torch.randn(3, 49, 4, 4)
