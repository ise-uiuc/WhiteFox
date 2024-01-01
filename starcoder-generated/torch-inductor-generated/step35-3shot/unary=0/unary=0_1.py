
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_0 = torch.nn.Conv2d(4, 2, (1, 3), stride=(2, 1), padding=[[0, 0], [1, 1]])
        self.conv2d_1 = torch.nn.Conv2d(4, 6, 2, stride=(3, 1), padding=1)
    def forward(self, x1):
        v1 = self.conv2d_0(x1)
        v2 = self.conv2d_1(x1)
        v3 = self.conv2d_0(v1)
        v4 = self.conv2d_0(v2)
        v5 = self.conv2d_0(v3)
        v6 = self.conv2d_0(v4)
        v7 = self.conv2d_1(v5)
        v8 = self.conv2d_0(v7)
        v9 = self.conv2d_0(v8)
        v10 = v5 * 0.044715
        v11 = v6 + v10
        v12 = v11 * 0.7978845608028654
        v13 = torch.tanh(v12)
        v14 = v13 + 1
        v15 = v14 * 0.5
        v16 = v14 * v14
        v17 = v17 * v14
        v18 = v18 * 0.044715
        v19 = v16 + v18
        v20 = v19 * 0.7978845608028654
        v21 = torch.tanh(v20)
        v22 = v21 + 1
        v23 = v15 * v22
        v24 = v17 * v22
        v25 = v24 * 0.044715
        v26 = v23 + v25
        return torch.cat((v26, v23, v26), 1)
# Inputs to the model
x1 = torch.randn(1, 4, 40)
