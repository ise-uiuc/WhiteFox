
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_1 = torch.nn.Conv2d(128, 64, 3, stride=2, padding=1)
        self.maxpool_1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn = torch.nn.BatchNorm2d(64, momentum=0.5)
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        v11 = self.maxpool(v10)
        v12 = self.conv_1(v11)
        v13 = v12 * 0.5
        v14 = v12 * v12
        v15 = v14 * v12
        v16 = v15 * 0.044715
        v17 = v12 + v16
        v18 = v17 * 0.7978845608028654
        v19 = torch.tanh(v18)
        v20 = v19 + 1
        v21 = v13 * v20
        v22 = self.maxpool_1(v21)
        v23 = v22.permute(0, 1, 4, 2, 3).contiguous()
        v24 = self.bn.forward(v23)
        return v24
# Inputs to the model
x2 = torch.randn(1, 64, 16, 16)
