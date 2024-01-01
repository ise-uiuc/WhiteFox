
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 480, 4, stride=2, padding=0)
        self.conv0 = torch.nn.Conv2d(3, 576, 4, stride=3, padding=0)
        self.conv1 = torch.nn.Conv2d(480, 576, 4, stride=1, padding=0)
        self.conv10 = torch.nn.Conv2d(3, 3, 4, stride=1, padding=0)
        self.conv11 = torch.nn.Conv2d(132, 3, 2, stride=3, padding=0)
    def forward(self, x9):
        v1 = self.conv(x9)
        v2 = self.conv0(x9)
        v3 = self.conv1(v2)
        v4 = v3 * 0.5
        v5 = v3 * v3
        v6 = v5 * v3
        v7 = v6 * 0.044715
        v8 = v3 + v7
        v9 = v8 * 0.7978845608028654
        v10 = torch.tanh(v9)
        v11 = v10 + 1
        v12 = v3 - v11
        v13 = math.sin(v12)
        v14 = torch.cat([v1, v13], 1)
        v15 = self.conv10(v14)
        v16 = v15 * 0.5
        v17 = v15 * v15
        v18 = v17 * v15
        v19 = v18 * 0.044715
        v20 = v15 + v19
        v21 = v20 * 0.7978845608028654
        v22 = torch.tanh(v21)
        v23 = v22 + 1
        v24 = v15 - v23
        v25 = math.sin(v24)
        v26 = torch.cat([v1, v15, v25], 1)
        v27 = v26 * 0.5
        v28 = v26 * v26
        v29 = v28 * v26
        v30 = v29 * 0.044715
        v31 = v26 + v30
        v32 = v31 * 0.7978845608028654
        v33 = torch.tanh(v32)
        v34 = v33 + 1
        v35 = v26 - v34
        v36 = math.sin(v35)
        return v36
# Inputs to the model
x9 = torch.randn(2, 3, 50, 70)
