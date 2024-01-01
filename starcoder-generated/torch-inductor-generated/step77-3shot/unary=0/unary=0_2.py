
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(435, 366, 3, stride=1, padding=2)
        self.conv_0 = torch.nn.BatchNorm2d(366)
        self.conv_1 = torch.nn.ReLU()
    def forward(self, x99967):
        v1 = self.conv(x99967)
        v2 = self.conv_0(v1)
        v3 = self.conv_1(v2)
        v4 = v3 * 0.5
        v5 = v3 * v3
        v6 = v5 * v3
        v7 = v6 * 0.044715
        v8 = v3 + v7
        v9 = v8 * 0.7978845608028654
        v10 = torch.tanh(v9)
        v11 = v10 + 1
        v12 = v4 * v11
        return v12
# Inputs to the model
x99967 = torch.randn(1, 435, 6, 33)
