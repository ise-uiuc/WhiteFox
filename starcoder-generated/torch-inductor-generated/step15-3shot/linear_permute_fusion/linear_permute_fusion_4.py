
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1, x2, x3, x4):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 1, 3, 2)
        v3 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
        v4 = v3.permute(0, 2, 3, 1)
        v5 = torch.nn.functional.linear(x3, self.linear.weight, self.linear.bias)
        v6 = torch.nn.functional.linear(x4, self.linear.weight, self.linear.bias)
        v7 = v6.permute(0, 1, 3, 2)
        v8 = torch.nn.functional.linear(x4, self.linear.weight, self.linear.bias)
        v9 = v8.permute(0, 2, 1, 3)
        return v2 + v4 + v7 + v9
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2, device='cpu')
x2 = torch.randn(1, 2, 2, 2, device='cpu')
x3 = torch.randn(1, 2, 2, 2, device='cpu')
x4 = torch.randn(1, 2, 2, 2, device='cpu')
