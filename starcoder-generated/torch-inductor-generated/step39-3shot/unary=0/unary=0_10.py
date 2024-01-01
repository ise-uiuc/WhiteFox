
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv5 = torch.nn.Conv2d(3, 63, 2, stride=2, padding=1)
        self.conv6 = torch.nn.Conv2d(63, 63, 2, stride=2, padding=1)
        self.conv7 = torch.nn.Conv2d(63, 63, 1, stride=1, padding=0)
        self.conv8 = torch.nn.Conv2d(63, 193, 1, stride=1, padding=0)
    def forward(self, x4):
        v1 = self.conv5(x4)
        v2 = self.conv6(v1)
        v3 = self.conv7(v2)
        v4 = v3 * 0.5
        v5 = v3 * v3
        v6 = v5 * v3
        v7 = v6 * 0.044715
        v8 = v3 + v7
        v9 = v8 * 0.7978845608028654
        v10 = torch.tanh(v9)
        v11 = v10 + 1
        v12 = v4 * v11
        v13 = v2 + v12
        v14 = self.conv8(v13)
        v15 = v14 * 0.5
        v16 = v14 * v14
        v17 = v16 * v14
        v18 = v17 * 0.044715
        v19 = v14 + v18
        v20 = v19 * 0.7978845608028654
        v21 = torch.tanh(v20)
        v22 = v21 + 1
        v23 = v15 * v22
        v24 = v13 + v23
        v25 = self.conv5(v24)
        v26 = self.conv6(v25)
        v27 = self.conv7(v26)
        v28 = v27 * 0.5
        v29 = v27 * v27
        v30 = v29 * v27
        v31 = v30 * 0.044715
        v32 = v27 + v31
        v33 = v32 * 0.7978845608028654
        v34 = torch.tanh(v33)
        v35 = v34 + 1
        v36 = v28 * v35
        v37 = v26 + v36
        v38 = self.conv5(v37)
        v39 = self.conv6(v38)
        v40 = self.conv7(v39)
        v41 = v40 * 0.5
        v42 = v40 * v40
        v43 = v42 * v40
        v44 = v43 * 0.044715
        v45 = v40 + v44
        v46 = v45 * 0.7978845608028654
        v47 = torch.tanh(v46)
        v48 = v47 + 1
        v49 = v41 * v48
        return v49
# Inputs to the model
x4 = torch.randn(1, 3, 400, 400)
