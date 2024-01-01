
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(3, 5, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(5, 7, 1, stride=1, padding=0)
    def forward(self, x11):
        v1 = self.conv(x11)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        v11 = self.conv1(x11)
        v12 = v11 * 0.5
        v13 = v11 * v11
        v14 = v13 * v11
        v15 = v14 * 0.044715
        v16 = v11 + v15
        v17 = v16 * 0.7978845608028654
        v18 = torch.tanh(v17)
        v19 = v18 + 1
        v20 = v12 * v19
        v21 = torch.add(v10, v20)
        v22 = self.conv2(v21)
        v23 = v22 * 0.5
        v24 = v22 * v22
        v25 = v24 * v22
        v26 = v25 * 0.044715
        v27 = v22 + v26
        v28 = v27 * 0.7978845608028654
        v29 = torch.tanh(v28)
        v30 = v29 + 1
        v31 = v23 * v30
        return v31
# Inputs to the model
x11 = torch.randn(3, 3, 223, 223)
