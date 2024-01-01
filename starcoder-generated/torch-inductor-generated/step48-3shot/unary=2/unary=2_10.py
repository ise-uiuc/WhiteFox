
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 4, (2, 2), stride=(1, 1), padding=(0, 0), dilation=1, groups=1, bias=False)
        self.batch_norm = torch.nn.BatchNorm2d(4, momentum=0.0010000000474974513, eps=0.0009999999747378752, affine=True, track_running_stats=True)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(4, 4, (3, 3), stride=(1, 1), padding=(0, 0), dilation=1, groups=1, bias=False)
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
        v10 = torch.sigmoid(v9)
        v11 = v10 + 1
        v12 = self.batch_norm(v11)
        v13 = self.conv_transpose2(v12)
        v14 = v13 * 0.5
        v15 = v13 * v13 * v13
        v16 = v15 * 0.044715
        v17 = v13 + v16
        v18 = v17 * 0.7978845608028654
        v19 = torch.tanh(v18)
        v20 = v19 + 1
        v21 = v14 * v20
        return v21
# Inputs to the model
x1 = torch.randn(3, 3, 5, 5)
