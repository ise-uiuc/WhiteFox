
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.21132487
        v3 = v1 * 0.11130278
        v4 = v1 * 0.22652966
        v5 = v1 * 0.028830857
        v6 = v1 * 0.2950478
        v7 = v1 * 0.37044453
        v8 = v2 * v1
        v9 = v3 * v1
        v10 = v4 * v1
        v11 = v5 * v1
        v12 = v6 * v1
        v13 = v7 * v1
        v14 = v8 - v9
        v15 = v10 - v11
        v16 = v11 - v12
        v17 = v12 - v13
        v18 = v13 - v1
        v19 = -v1 + v10
        v20 = -v1 - v10
        v21 = -v2 + v3
        v22 = -v2 - v3
        v23 = -v12 + v11
        v24 = v14 + v16
        v25 = v15 + v17
        v26 = v15 + v18
        v27 = v16 + v17
        v28 = v17 + v18
        v29 = v18 + v21
        v30 = v19 + v20
        v31 = -v19 + v20
        v32 = -v11 + v12
        v33 = v22 + v23
        v34 = v24 + v26
        v35 = -v24 - v26
        v36 = v25 + v27
        v37 = v25 + v28
        v38 = v27 + v28
        v39 = v28 + v30
        v40 = v29 + v34
        return v40
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
