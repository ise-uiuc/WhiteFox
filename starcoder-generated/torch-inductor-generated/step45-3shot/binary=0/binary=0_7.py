
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x1, **bnni):
        v1 = self.bn(x1)
        v2 = v1 + 0.1
        v3 = v2 + 0.1
        v4 = v3 * 0.1
        v5 = v4 * 0.1
        v6 = torch.zeros(v5.shape)
        v7 = v6 + 0.1
        v8 = v7 * 0.1
        if "test" in bnni:
            v9 = bnni["test"] + v8
        else:
            v9 = v8
        return v9
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
