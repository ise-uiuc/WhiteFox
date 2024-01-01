
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose3d(3, 12, kernel_size=1, stride=(1, 2, 3))
        self.conv_transpose2 = torch.nn.ConvTranspose3d(3, 12, kernel_size=3, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(x1)
        v3 = v1 * 0.5
        v4 = v1 * v1 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64, 64)
