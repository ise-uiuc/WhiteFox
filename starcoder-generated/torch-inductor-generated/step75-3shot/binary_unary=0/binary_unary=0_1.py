
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3, x4, x5, x6):
        v1 = self.conv1(x1)
        v2 = v1 + x2
        a1 = torch.tanh(v2)
        v3 = self.conv2(x2)
        v4 = torch.nn.ReLU()(v2)
        v5 = v3 + x3
        a2 = torch.nn.Sigmoid()(v5)
        v6 = torch.nn.ReLU()(a1 + v5)
        v7 = self.conv3(x4)
        v8 = v7 + a2
        v9 = torch.nn.ReLU()(v8)
        v10 = self.conv2(x5 + v9)
        a3 = torch.nn.Tanh()(v10)
        v11 = torch.nn.ReLU()(v10)
        v12 = torch.nn.ReflectionPad2d(1)(a3)
        v13 = self.conv3(v12)
        a4 = torch.nn.MaxPool2d(3, padding=1, dilation=1)(v11)
        v14 = v13 + a4
        v15 = torch.nn.ReLU()(v14)
        v16 = self.conv1(x6 + v15)
        a5 = torch.nn.Tanh()(v16)
        v17 = torch.nn.MaxPool2d(3, padding=1, dilation=1)(v11)
        v18 = torch.nn.ReLU()(a5 + v17)
        v19 = torch.nn.Conv2d(16, 16, 3, bias=False, stride=2, padding=1)(v18)
        a6 = torch.nn.Tanh()(v19)
        v20 = torch.nn.MaxPool2d(3, padding=1, dilation=1)(v18)
        v21 = torch.nn.ReLU()(a6 + v20)
        v22 = torch.nn.Conv2d(16, 16, 3, bias=False, stride=2, padding=1)(v21)
        v23 = torch.nn.ReLU()(v22)
        return v23
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
x5 = torch.randn(1, 16, 64, 64)
x6 = torch.randn(1, 16, 64, 64)
