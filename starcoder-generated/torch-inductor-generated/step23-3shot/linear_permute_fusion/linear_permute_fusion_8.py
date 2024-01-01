
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1, x2, x3):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.permute(0, 3, 2, 1)
        v4 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
        v5 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        v6 = v5.permute(0, 3, 2, 1)
        v7 = torch.nn.functional.linear(x3, self.linear.weight, self.linear.bias)
        v8 = torch.nn.functional.linear(v7, self.linear.weight, self.linear.bias)
        v9 = v8.permute(0, 2, 1, 3)
        return v3 + v6 + v9
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2, device='cpu')
x2 = torch.randn(1, 2, 2, 2, device='cpu')
x3 = torch.randn(1, 2, 2, 2, device='cpu')
