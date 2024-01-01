
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(16, 16, kernel_size=1, stride=1, bias=True)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.nn.Softmax(dim=0)(torch.cat((v1, v1)))
        v3 = v2.max()[0]
        v4 = v2.min()[0]
        v5 = v3 + v4
        v6 = v5 * v4
        v7 = v6 * v5
        return v7
# Inputs to the model
x1 = torch.randn(10, 16, 100)
