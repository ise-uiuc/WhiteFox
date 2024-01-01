
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(1, 1, (1, 1), stride=1, padding=(0, 0))
        self.conv_2 = torch.nn.Conv2d(1, 1, (1, 1), stride=1, padding=(0, 0))
    def forward(self, x):
        negative_slope = 0.0
        v1 = self.conv_1(x)
        v2 = self.conv_2(x)
        v3 = v1 > 0
        v4 = v2 > 0
        v5 = (v3, v1, v2)
        v6 = torch.where(v3, v1, v2)
        v7 = [i for i in v5]
        v8 = [i for i in v7]
        v9 = v4 == v8
        v10 = v3 == v4
        return v10
# Inputs to the model
x1 = torch.randn(1, 1, 1, 1)
