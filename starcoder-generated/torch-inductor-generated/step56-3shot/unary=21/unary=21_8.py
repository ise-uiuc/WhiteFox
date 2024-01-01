
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv_2 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv_3 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv_4 = torch.nn.Conv2d(16, 32, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv_5 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv_6 = torch.nn.Conv2d(32, 64, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv_7 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
    def forward(self, x):
        v1 = self.conv_1(x)
        v2 = torch.tanh(v1)
        v3 = v2
        v4 = v3
        v5 = self.conv_2(v4)
        v6 = torch.tanh(v5)
        v7 = v6
        v8 = v7
        v9 = self.conv_3(v8)
        v10 = torch.tanh(v9)
        v11 = v10
        v12 = v11
        v13 = self.conv_4(v12)
        v14 = torch.tanh(v13)
        v15 = v14
        v16 = v15
        v17 = v16
        v18 = torch.mul(v17, v1)
        v19 = self.conv_5(v18)
        v20 = torch.tanh(v19)
        v21 = v20
        v22 = v21
        v23 = self.conv_6(v22)
        v24 = torch.tanh(v23)
        v25 = v24
        v26 = v25
        v27 = v26
        v28 = v27
        v29 = torch.mul(v28, v5)
        v30 = self.conv_7(v29)
        v31 = torch.tanh(v30)
        v32 = v31
        v33 = v32
        return v33
# Inputs to the model
x = torch.randn(1, 16, 56, 56)
