
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(12, 21, 1, stride=1, padding=0)
        self.conv_0 = torch.nn.Conv2d(12, 21, 1, stride=1, padding=0)
        self.conv_1 = torch.nn.Sigmoid()
        self.conv_2 = torch.nn.Conv2d(3, 4, 2, stride=1, padding=0)
        self.conv_3 = torch.nn.Conv2d(4, 5, 1, stride=1, padding=1)
    def forward(self, x12):
        v1 = self.conv(x12)
        v2 = self.conv_0(x12)
        v3 = self.conv_1(x12)
        v8 = self.conv_2(v3)
        v4 = v1 * 0.5
        v5 = v1 * v1
        v6 = v5 * v1
        v7 = v6 * 0.044715
        v121 = v1 + v7
        v9 = v121 * 0.7978845608028654
        v10 = torch.tanh(v9)
        v11 = v10 + 1
        v12 = v2 * v11
        v13 = v12 * v8
        v14 = v13 * 0.5
        return v14
# Inputs to the model
x12 = torch.randn(1, 12, 12, 12)
