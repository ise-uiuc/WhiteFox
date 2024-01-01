
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 3, stride=2, padding=0, dilation=1)
        self.conv2 = torch.nn.Conv2d(6, 12, 3, stride=2, padding=0, dilation=1)
        self.batchnorm = torch.nn.BatchNorm2d(num_features=12)
    def forward(self, x1):
        x = torch.nn.Hardsigmoid()(x1)
        v1 = self.conv1(x)
        v2 = torch.nn.Hardtanh()(v1)
        v3 = self.conv2(v2)
        v4 = torch.nn.Hardtanh()(v2)
        v5 = self.batchnorm(v4)
        v6 = torch.nn.AvgPool2d(3)(v5)
        v7 = torch.nn.AvgPool2d(3)(v6)
        v8 = torch.norm(v5)
        v9 = torch.norm(v7)
        result = v8 + v9
        v10 = torch.norm(v9)
        v11 = (v10)
        v12 = torch.max(v11, v10)
        v13 = torch.max(v12, 0)
        v14 = torch.max(v13, 0)
        v15 = torch.gather(v10, 0, -2)
        v16 = torch.gather(v14, -1, -1)
        v17 = torch.gather(v14, -2, -3)
        v18 = torch.max(v11, v15)
        v19 = torch.max(v18, 0)
        v20 = torch.max(v19, 0)
        v21 = torch.gather(v15, 0, -2)
        v22 = torch.gather(v20, -1, -1)
        v23 = torch.gather(v20, -2, -3)
        v24 = torch.max(v11, v17)
        v25 = torch.max(v24, 0)
        v26 = torch.max(v25, 0)
        v27 = torch.gather(v17, 0, -2)
        v28 = torch.gather(v26, -1, -1)
        v29 = torch.gather(v26, -2, -3)
        result0 = v21 + v22
        result1 = v23 + result0
        result2 = v27 + v28
        result3 = v29 + result2
        result4 = v16 + result1
        result5 = v7 + result4
        result6 = v10 + v2
        result7 = result3 + result6
        return result7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
