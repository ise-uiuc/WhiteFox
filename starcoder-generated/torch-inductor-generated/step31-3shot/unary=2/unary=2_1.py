
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(64, 16, 4)
        self.conv2d = torch.nn.Conv2d(16, 8, 4)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv2d(v1)
        v3 = v2 * 0.5
        v4 = v2 * v2 * v2
        v5 = v2 + v4
        v6 = v5 * 0.044715
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v3 * v8
        return v9
# Inputs to the model
x1 = torch.randn(3, 64, 16, 16)
