
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(4, 16, 3, stride=3, padding=0)
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
        v10 = torch.nn.MaxPool3d(kernel_size=(1, 1, 1), stride=(1, 1, 1))(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 4, 8, 3, 3)
