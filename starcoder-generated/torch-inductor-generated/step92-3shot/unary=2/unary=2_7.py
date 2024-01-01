
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(11, 1, 1, stride=2, padding=2)
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=2, output_padding=0)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        v10 = self.conv_transpose(v9)
        v11 = self.conv_transpose2(v10)
        return v11
# Inputs to the model
x1 = torch.randn(3, 11, 64, 254)
