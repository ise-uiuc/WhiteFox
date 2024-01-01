
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 9, (10, 30), stride=(5, 10), padding=(2, 4))
        self.conv2 = torch.nn.Conv2d(9, 36, (10, 20), stride=(5, 10), padding=(2, 4))
        self.conv3 = torch.nn.Conv2d(36, 10, (30, 20), stride=(5, 3), padding=(4, 6))
    def forward(self, x1):
        v2 = self.conv1(x1)
        v4 = v2 * 0.5
        v5 = v2 * v2
        v6 = v5 * v2
        v7 = v6 * 0.044715
        v8 = v2 + v7
        v9 = v8 * 0.7978845608028654
        v10 = torch.tanh(v9)
        v11 = v10 + 1
        v12 = v4 * v11
        v13 = self.conv2(v12)
        v15 = v13 * 0.5
        v16 = v13 * v13
        v17 = v16 * v13
        v18 = v17 * 0.044715
        v19 = v13 + v18
        v20 = v19 * 0.7978845608028654
        v21 = torch.tanh(v20)
        v22 = v21 + 1
        v23 = v15 * v22
        v24 = self.conv3(v23)
        v26 = v24 * 0.5
        v27 = v24 * v24
        v28 = v27 * v24
        v29 = v28 * 0.044715
        v30 = v24 + v29
        v31 = v30 * 0.7978845608028654
        v32 = torch.tanh(v31)
        v33 = v32 + 1
        v35 = v33 * v26
        return v35
# Inputs to the model
x1 = torch.randn(1, 3, 32, 52)
