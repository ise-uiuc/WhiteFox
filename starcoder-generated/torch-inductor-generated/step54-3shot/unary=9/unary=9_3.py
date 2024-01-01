
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 1, stride=2, padding=1)
        self.bias_conv = torch.nn.Conv2d(8, 8, 1, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = v2.clamp(min=0, max=6)
        v4 = 10 + v3
        v5 = v4.clamp(max=11)

        v6 = v3.div(6)
        v7 = self.other_conv(v6)
        v8 = 3 + v7
        v9 = v8.clamp(min=0, max=6)
        v10 = v9 / 6

        self.bias_conv.bias.data = torch.clamp(7, min=0, max=10)
        v11 = self.bias_conv(v8)
        v12 = self.bias_conv.bias.data
        v13 = v12 * 2
        v14 = 3 + v13
        v15 = v14.clamp(min=0, max=6)
        v16 = 2 + v15
        v17 = v16 / 6
        return v10
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
