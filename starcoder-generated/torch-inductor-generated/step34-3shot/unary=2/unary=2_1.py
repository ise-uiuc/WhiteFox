
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(1, 2, 1, stride=2, padding=2, dilation=2, groups=1, bias=False)
        self.conv3 = torch.nn.ConvTranspose2d(6, 11, 11, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.conv4 = torch.nn.ConvTranspose2d(15, 20, 20, stride=1, padding=0, dilation=1, groups=1, bias=False)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        v10 = self.conv3(x2)
        v11 = v10 * 0.5
        v12 = v10 * v10 * v10
        v13 = v12 * 0.044715
        v14 = v10 + v13
        v15 = v14 * 0.7978845608028654
        v16 = torch.tanh(v15)
        v17 = v16 + 1
        v18 = v11 * v17
        v19 = v9 + v18
        v20 = v19 * 0.7978845608028654
        v21 = torch.tanh(v20)
        v22 = v21 + 1
        v23 = self.conv4(x1)
        v24 = v23 * 0.5
        v25 = v23 * v23 * v23
        v26 = v25 * 0.044715
        v27 = v23 + v26
        v28 = v27 * 0.7978845608028654
        v29 = torch.tanh(v28)
        v30 = v29 + 1
        v31 = v24 * v30
        v32 = v31 + v22
        v33 = v32 * 0.7978845608028654
        v34 = torch.tanh(v33)
        v35 = v34 + 1
        v36 = v2 * v35
        return v36
# Inputs to the model
x1 = torch.randn(10, 1, 100, 150)
x2 = torch.randn(10, 6, 90, 135)
