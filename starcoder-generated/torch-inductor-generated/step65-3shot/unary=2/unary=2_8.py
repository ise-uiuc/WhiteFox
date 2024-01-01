
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(4, 5, 3, stride=(2), padding=1)
        self.conv2d = torch.nn.Conv2d(1, 10, 3, stride=(1), padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(2, 3, 3, stride=(2), padding=1)
        self.conv_transpose3 = torch.nn.ConvTranspose3d(2, 3, 3, stride=(1, 1, 1), padding=0)
    def forward(self, x1, x2):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        v10 = self.conv2d(v9)
        v11 = self.conv_transpose2(x2)
        v12 = self.conv_transpose3(v11)
        return v10, v12
# Inputs to the model
x1 = torch.randn(3, 4, 3)
x2 = torch.randn(1, 2, 3, 3, 3)
