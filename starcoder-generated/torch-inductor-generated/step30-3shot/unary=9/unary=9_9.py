
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(16, 8, 16, stride=16)
    def forward(self, x1):
        v1 = self.conv2(x1)
        v1_0 = 0.0001 + v1
        v1_1 = -0.0001 + v1_0
        v1_2 = -0.0001 + v1_1
        v2 = -0.0001 + v1_2
        v3 = 3.14 + v2
        v4 = v3 - 3.14
        v5 = v4 + 3.14
        v6 = v5 + v5
        v7 = v6 - 2 * v4
        v8 = v7 / 2
        v9 = v8.div(3)
        v10 = -0.1 + v9
        v11 = v10.clamp(min=0, max=6)
        v12 = v11/6
        return v12
# Inputs to the model
x1 = torch.randn(15, 16, 128, 128)
