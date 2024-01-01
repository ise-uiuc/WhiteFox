
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_a = torch.nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1, output_padding=1)
        self.conv_transpose_b = torch.nn.ConvTranspose2d(16, 12, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_a(x1)
        v2 = self.conv_transpose_b(x1)
        v3 = torch.tanh(v1)
        v4 = torch.tanh(v2)
        v5 = v3 + v4
        return v1, v5
# Inputs to the model
x1 = torch.randn(1, 16, 26, 26)
