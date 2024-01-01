
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(14, 9, 7, stride=1, padding=3, dilation=2, groups=2)
        self.conv1 = torch.nn.ConvTranspose2d(9, 12, 4, stride=2, padding=1, output_padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(12, 10, 3, stride=2, padding=0, output_padding=1)
        self.conv3 = torch.nn.Conv2d(10, 11, 5, stride=1, padding=2, dilation=2, groups=1)
        self.conv4 = torch.nn.Conv2d(15, 5, 3, stride=1, padding=1, groups=2)
        self.conv5 = torch.nn.Conv2d(7, 3, 5, stride=1, padding=1, groups=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sum(v1, dim=1)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        v8 = self.conv1(v7)
        v9 = v8 * 0.5
        v10 = v8 * 0.7071067811865476
        v11 = torch.erf(v10)
        v12 = v11 + 1
        v13 = v9 * v12
        v14 = self.conv2(v13)
        v15 = v14 * 0.5
        v16 = v14 * 0.7071067811865476
        v17 = torch.erf(v16)
        v18 = v17 + 1
        v19 = v15 * v18
        v20 = self.conv3(v19)
        v21 = torch.mean(v20, dim=[0, 2, 3])
        v22 = self.conv4(x1)
        v23 = v22 * 0.5
        v24 = v22 * 0.7071067811865476
        v25 = torch.erf(v24)
        v26 = v25 + 1
        v27 = v23 * v26
        v28 = self.conv5(x1)
        v29 = v28 * 0.5
        v30 = v28 * 0.7071067811865476
        v31 = torch.erf(v30)
        v32 = v31 + 1
        v33 = v29 * v32
        return v21 * v27, v33
# Inputs to the model
x1 = torch.randn(1, 14, 30, 30)
