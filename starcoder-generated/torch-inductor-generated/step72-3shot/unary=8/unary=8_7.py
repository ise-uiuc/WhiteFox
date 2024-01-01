
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, 3, stride=2, padding=1, output_padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 2.5
        v3 = v1 + v2
        v4 = torch.clamp(v3, min=-1)
        v5 = v4 * v4
        v6 = v5.permute(0, 2, 1)
        v7 = v3.matmul(v6)
        v8 = v1.pow(2)
        v9 = v3.sin()
        v10 = v1.tanh()
        v11 = v1.relu()
        v12 = v1 + v9
        return v12
# Inputs to the model
x1 = torch.randn(4, 3, 28, 28)
