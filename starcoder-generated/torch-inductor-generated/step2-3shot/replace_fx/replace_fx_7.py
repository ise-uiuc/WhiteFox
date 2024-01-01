
class SubModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = torch.nn.Parameter(torch.randn(2, 2))
    def forward(self, input):
        return input + self.x
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = SubModule()
        self.dropout1 = torch.nn.Dropout(0.8)
        self.dropout2 = torch.nn.Dropout(0.3)
    def forward(self, x, y):
        v1 = x * y
        v2 = self.l1(x)
        v3 = v1 + v2
        v4 = torch.matmul(v1, v2)
        v5 = self.dropout1(v3)
        v6 = v4 + v5
        v7 = self.dropout2(v6)
        return v7.sum()
# Inputs to the model
x = torch.randn(10)
y = torch.randn(10)
