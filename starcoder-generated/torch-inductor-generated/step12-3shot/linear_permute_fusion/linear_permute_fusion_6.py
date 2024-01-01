
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1, x2):
        v10 = x2
        v9 = v10
        v8 = v9
        v7 = v8
        v6 = v7
        v5 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v4 = v5.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        v2 = v3.permute(0, 2, 1)
        v1 = v2.permute(0, 2, 1)
        v11 = v1
        return v11
# Inputs to the model
x1 = torch.randn(3, 2, 2)
x2 = torch.randn(3, 2, 2)
