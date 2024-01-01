
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1, x2, x3, x4):
        v3 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
        v4 = v3.permute(0, 2, 1)
        v6 = torch.nn.functional.linear(x4, self.linear.weight, self.linear.bias)
        v7 = v6.permute(0, 2, 1)
        v1 = x1
        v2 = torch.nn.functional.linear(x3, self.linear.weight, self.linear.bias)
        v5 = v2.permute(1, 0, 2, 3)
        v8 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        return v5 + v4 + v7 + v8
# Inputs to the model
x1 = torch.randn(2, 1, 2)
x2 = torch.randn(1, 2, 2, 2, device='cpu')
x3 = torch.randn(1, 2, 2, 2, device='cpu')
x4 = torch.randn(1, 2, 2, 2, device='cpu')
