
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
        v4 = v3.permute(0, 2, 1)
        v5 = torch.nn.functional.linear(x3, self.linear.weight, self.linear.bias)
        v6 = torch.nn.functional.linear(x4, self.linear.weight, self.linear.bias)
        v7 = v6.permute(0, 2, 1)
        v8 = torch.nn.functional.linear(x5, self.linear.weight, self.linear.bias)
        v10 = v8.permute(0, 2, 1)
        v9 = torch.nn.functional.linear(x6, self.linear.weight, self.linear.bias)
        v11 = torch.nn.functional.linear(x7, self.linear.weight, self.linear.bias)
        v12 = v11.permute(0, 2, 1)
        v13 = torch.nn.functional.linear(x8, self.linear.weight, self.linear.bias)
        v14 = torch.nn.functional.linear(x9, self.linear.weight, self.linear.bias)
        v15 = v14.permute(0, 2, 1)
        return v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10 + v11 + v12 + v13 + v14 + v15
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
x3 = torch.randn(1, 2, 2)
x4 = torch.randn(1, 2, 2)
x5 = torch.randn(1, 2, 2)
x6 = torch.randn(1, 2, 2)
x7 = torch.randn(1, 2, 2)
x8 = torch.randn(1, 2, 2)
x9 = torch.randn(1, 2, 2)
