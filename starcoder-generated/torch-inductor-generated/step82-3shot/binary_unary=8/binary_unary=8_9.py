
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pointwise_conv1 = torch.nn.Conv2d(1, 10, (11, 2), stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.conv2 = torch.nn.Conv2d(10, 9, (5, 1), stride=1, padding=0, dilation=1, groups=1, bias=True)
    def forward(self, x1):
        v1 = self.pointwise_conv1(x1)
        v2 = self.pointwise_conv1(x1)
        v3 = self.conv2(v1)
        v4 = self.conv2(v1)
        v5 = self.conv2(v2)
        v6 = self.conv2(v2)
        v7 = v3 + v4
        v8 = torch.relu(v3)
        v9 = self.conv2(v1)
        v10 = torch.conv2d(v5)
        v11 = self.conv2(v5)
        v12 = self.conv2(v6)
        v13 = self.conv2(v6)
        v14 = v9 + v10
        v15 = torch.relu(v9)
        v16 = torch.conv2d(v11)
        v17 = self.conv2(v11)
        v18 = self.conv2(v12)
        v19 = self.conv2(v12)
        v20 = v16 + v17
        v21 = torch.relu(v16)
        v22 = torch.conv2d(v18)
        v23 = self.conv2(v18)
        v24 = self.pointwise_conv1(x1)
        v25 = torch.conv2d(x2)
        v26 = torch.conv2d(x2)
        v27 = v22 + v23
        v28 = torch.relu(v22)
        v29 = self.conv2(v18)
        v30 = self.conv2(v24)
        v31 = self.conv2(v30)
        v32 = self.conv2(v31)
        v33 = torch.relu(v27)
        v34 = self.conv2(v30)
        v35 = self.conv2(v25)
        v36 = torch.conv2d(v26)
        v37 = self.conv2(v26)
        v38 = self.conv2(v34)
        v39 = self.conv2(v35)
        v40 = torch.relu(v33)
        return v40
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
x2 = torch.randn(1, 1, 64, 64)
