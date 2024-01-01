
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 4, (1, 1), stride=(1, 1), bias=False)
    def forward(self, x0):
        v1 = x0.transpose(-1, -2).reshape(-1, 1, 3, 4).contiguous()
        v2 = self.conv_transpose(v1)
        v3 = v2 * 0.5
        v4 = v2 * v2 * v2
        v5 = v4 * 0.044715
        v6 = v2 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v3 * v8
        v11 = v10.view(5, 3, 4).transpose(-1, -2).reshape(5, -1)
        return v11
# Inputs to the model
x0 = torch.randn(5, 10, 2)
