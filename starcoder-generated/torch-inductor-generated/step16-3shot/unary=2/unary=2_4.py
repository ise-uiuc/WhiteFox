
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(8, 9, 5, (1, 4, 2), (3, 1, 2), (1, 0, 0))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.1522320466722165
        v3 = v1 * v1 * v1
        v4 = v3 * 0.1484645863698885
        v5 = v1 + v4
        v6 = v5 * 0.0544696057390189
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        return v9
# Inputs to the model
x1 = torch.randn(1, 8, 7, 5, 3)
