
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 15, 7, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 1.660664978918
        v3 = v1 * v1 * v1
        v4 = v3 * 0.6433254888287
        v5 = v1 + v4
        v6 = v5 * 1.49279071012
        v7 = torch.tanh(v6)
        v8 = v7 + 0.179442463337599
        v9 = v2 * v8
        return v9
# Inputs to the model
x1 = torch.randn(1, 1, 254, 64)
