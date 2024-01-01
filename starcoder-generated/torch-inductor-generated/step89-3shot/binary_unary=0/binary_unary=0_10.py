
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 1024, bias=True)
        self.linear1 = torch.nn.Linear(1024, 2048, bias=True)
        self.linear2 = torch.nn.Linear(2048, 2048, bias=True)
        self.linear3 = torch.nn.Linear(2048, 2048, bias=True)
    def forward(self, x1, x2, x3, x4):
        v1 = self.linear(x1)
        v2 = self.linear(x2)
        v3 = self.linear(x3)
        v4 = self.linear(x4)
        v5 = v1 + v2
        v6 = self.linear1(v5)
        v7 = self.linear2(v6)
        v8 = v7 + v3
        v9 = self.linear3(v8)
        v10 = v9 + v4
        v11 = v10 + v2
        return v11
# Inputs to the model
x1 = torch.randn(1024)
x2 = torch.randn(1024)
x3 = torch.randn(1024)
x4 = torch.randn(1024)
