
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 11, bias=False, pad_mode='pad', padding=(9, 10), stride=(1, 1), dilation=(1, 2))
        self.conv2 = torch.nn.Conv2d(8, 2, 11, bias=False, pad_mode='pad', padding=(8, 9), stride=(1, 52), dilation=(1, 1), groups=1)
        self.conv3 = torch.nn.Conv2d(2, 5, 1, bias=False, pad_mode='pad', padding=(5, 3), stride=(35, 1), dilation=(1, 2))
        self.conv4 = torch.nn.Conv2d(5, 4, 1099, bias=False, pad_mode='pad', padding=(17, 14), stride=(1, 52), dilation=(1, 3))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = v7 * 0.5
        v9 = v7 * 0.7071067811865476
        v10 = torch.erf(v9)
        v11 = v10 + 1
        v12 = v8 * v11
        v13 = self.conv3(v12)
        v14 = v13 * 0.5
        v15 = v13 * 0.7071067811865476
        v16 = torch.erf(v15)
        v17 = v16 + 1
        v18 = v14 * v17
        v19 = self.conv4(v18)
        return v19
# Inputs to the model
x1 = torch.randn(1, 3, 1099, 287)
