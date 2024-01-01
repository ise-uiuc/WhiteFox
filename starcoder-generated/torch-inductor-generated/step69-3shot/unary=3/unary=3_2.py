
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(9, 33, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(33, 39, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * -1
        v3 = v2 * -1
        v4 = v3 * -1
        v5 = v4 * -1
        v6 = v5 * -1
        v7 = v1 * 5
        v8 = v3 * 5
        v9 = v4 * -3
        v10 = v6 * -3
        v11 = v8 * -3
        v12 = v10 * -3
        v13 = v11 * -3
        v14 = v12 * 5
        v15 = v7 + v13
        v16 = v14 + v9
        v17 = self.conv2(v16)
        return v17
# Inputs to the model
x1 = torch.randn(5, 9, 224, 224)
