
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1
        v3 = v1
        v4 = v2 + v1
        v5 = v4
        v6 = v4
        v7 = v4 / v6
        v8 = v4
        v9 = v4
        v10 = v9
        v11 = v6 + v10
        v12 = v5 + v11
        v13 = v11
        v14 = v12 + v13
        return v11
# Inputs to the model
x1 = torch.randn(1, 5, 2)
