
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.BatchNorm2d(19)
        self.conv_0 = torch.nn.ReLU()
        self.conv_1 = torch.nn.Conv2d(19, 47, 1, stride=1, padding=0)
    def forward(self, x958):
        v1 = self.conv(x958)
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
x958 = torch.randn(1, 19, 95, 69)
