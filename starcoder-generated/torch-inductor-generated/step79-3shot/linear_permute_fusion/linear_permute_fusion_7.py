
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x2):
        v1 = torch.sum(x2, dim=2)
        v2 = v1 / 3.1415926
        v3 = v1 * 3.1415926
        v4 = v2 + v3
        v5 = 3.1415926 + v1
        v6 = v2 + v2
        v7 = v3 - v2
        v8 = v2 / 7.8539816
        v9 = v8 / 4.71238898
        v10 = v8 - v9
        v11 = v5 * v10
        return v4 / v11
# Inputs to the model
x2 = torch.randn(1, 2, 2)
