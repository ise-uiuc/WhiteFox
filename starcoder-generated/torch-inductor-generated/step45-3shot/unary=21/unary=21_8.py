
class PatternModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(100, 112, 11, stride=1, padding=0, bias=False)
        self.conv3 = torch.nn.Conv2d(80, 249, 13, stride=1, padding=7, bias=True)
        self.conv31 = torch.nn.Conv2d(31, 118, 1, stride=1, padding=0, bias=False)
        self.add1 = torch.nn.quantized.FloatFunctional()
        self.sigmoid = torch.nn.Sigmoid()
        self.conv4 = torch.nn.Conv2d(223, 69, 20, stride=1, padding=0, bias=True)
        self.tanh = torch.nn.Tanh()
    def forward(self, x3):
        v3 = self.conv1(x3)
        v4 = self.conv3(x3)
        v6 = self.conv31(v3)
        v5 = self.add1.add(v4, v6)
        v7 = self.sigmoid(v4)
        v8 = v4.permute(0, 1, 3, 2)
        v81 = v8.permute(0, 3, 1, 2)
        v9 = v5
        v8 = self.conv4(v81)
        v16 = v9 + v8
        v8 = v8.reshape(83*2048*2048)
        v8 = v8.reshape(2048, 2048, 83)
        v81 = v8.permute(2, 1, 0)
        v17 = v81.reshape(1664*2048)
        v16 = self.tanh(v16)
        return v16
# Inputs to the model
x3 = torch.randn(1, 100, 2048, 2048)
