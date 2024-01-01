
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
        self.sigmoid = torch.nn.ReLU()
    def forward(self, x1):
        v1 = x1 + 1
        v2 = x1 + 2
        v3 = v1.permute(1, 0)
        v4 = v2.permute(1, 0)
        v5 = v3.permute(1, 0)
        v6 = torch.cat((v3, v4), dim = -1)
        v7 = torch.cat((v5, v6), dim = -1)
        v8 = self.linear(v7)
        v9 = v7.permute(1, 0)
        v10 = v9.permute(1, 0)
        v11 = torch.cat((v9, v10), dim = -1)
        v12 = torch.cat((v11, v8), dim = -1)
        v13 = v12.permute(1, 0)
        return v13
# Inputs to the model
x1 = torch.randn(1, 4, 2)
