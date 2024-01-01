
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(7, 4, (1, 1), stride=(1, 1))
        self.conv_transpose2 = torch.nn.ConvTranspose2d(4, 2, (3, 3), stride=(2, 2))
        self.conv_transpose3 = torch.nn.ConvTranspose2d(2, 1, (5, 5), stride=(1, 1))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.078835
        v3 = v1 + v2
        v4 = v3 * 0.085504
        v5 = torch.tanh(v4)
        v6 = v5 * 0.836325
        v7 = torch.tanh(v6)
        v8 = v7 * 0.330093
        v9 = self.conv_transpose2(v8)
        v10 = v9 * 0.275604
        v11 = v9 + v10
        v12 = v11 * 0.945549
        v13 = torch.tanh(v12)
        v14 = torch.tanh(v13)
        v15 = v14 * 0.276435
        v16 = self.conv_transpose3(v15)
        v17 = v16 + 0.5816172726678164
        return v17
# Inputs to the model
x1 = torch.randn(3, 7, 2, 2)
